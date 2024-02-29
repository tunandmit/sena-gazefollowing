import ast
import torch
import math
import numpy as np
from typing import List
from .data_utils import get_label_map


def convert_str_to_list(x):
    return ast.literal_eval(x)

def prepare_finetune_format(example, gaze_heatmap_size: int=64):
    num_bboxes = len(example['labels'])
    if example['gazeIdx'] == -1:
      target = {
        'regression_padding' : torch.full((num_bboxes, 1), True),
        'gaze_point' : torch.full((num_bboxes, 1, 2), 0).float(),
        # 'gaze_vector' : torch.full((num_bboxes, 1, 2), 0), 
        'gaze_watch_outside' : torch.BoolTensor([True]).long(),
      }
    else:
      eye = [float(example['hx']) / 640, float(example['hy']) / 480]
      gaze = [float(example['gaze_cx']) / 640, float(example['gaze_cy']) / 480]

      eye = torch.FloatTensor([eye[0], eye[1]])
      tgt_gaze = torch.FloatTensor([gaze[0], gaze[1]]).view(1,2)

      # gaze_vector = get_angle_magnitude(eye, tgt_gaze)

      target = {
        'gaze_point' : tgt_gaze,
        # 'gaze_vector' : gaze_vector,
        'gaze_watch_outside' : torch.BoolTensor([False]).long()
      }
      # represents which objects have a gaze heatmap
      regression_padding = torch.full((num_bboxes, 1), True)
      regression_padding[: len(target["gaze_point"])] = False
      target['regression_padding'] = regression_padding

      gaze_points = torch.full((num_bboxes, 1, 2), 0).float()
      gaze_points[: len(target['gaze_point']), :, :] = target["gaze_point"]
      target['gaze_point'] = gaze_points

      # gaze_vectors = torch.full((num_bboxes, 1, 2), 0).float()
      # gaze_vectors[: len(target['gaze_vector']), :] = target['gaze_vector']
      # target['gaze_vector'] = gaze_vectors

    gaze_watch_outside = torch.full((num_bboxes, 1), 0).float()
    gaze_watch_outside[: len(target["gaze_watch_outside"]), 0] = target["gaze_watch_outside"]
    target["gaze_watch_outside"] = gaze_watch_outside.long()

    gaze_heatmaps = []
    for gaze_point, regression_padding in zip(
        target["gaze_point"], target["regression_padding"]
    ):
      gaze_x, gaze_y = gaze_point.squeeze(0)
      
      if regression_padding:
        gaze_heatmap = torch.full(
          (gaze_heatmap_size, gaze_heatmap_size),
          float(1.0)
        )
      else:
        gaze_heatmap = torch.zeros(
          (gaze_heatmap_size, gaze_heatmap_size)
        )

        gaze_heatmap = get_label_map(
            gaze_heatmap,
            [
              gaze_x * gaze_heatmap_size,
              gaze_y * gaze_heatmap_size,
            ],
            3,
            "Gaussian",
        )
      gaze_heatmaps.append(gaze_heatmap)
    
    target["gaze_heatmaps"] = torch.stack(gaze_heatmaps)

    return target
  
def prepare_evaluate_format(example, gaze_heatmap_size: int=64):
    num_queries = 300 # get default
    num_bbox = len(example['labels'])
    
    if example['gazeIdx'] == -1:
      
      target = {
        'regression_padding' : torch.full((num_queries, 1), True),
        'gaze_point' : torch.full((num_queries, num_bbox, 2), 0).float(),
        'gaze_watch_outside' : torch.BoolTensor([True]).long(),
        'gaze_points_padding' : torch.full((num_queries, num_bbox), True),
        "img_size": torch.FloatTensor([example['height'], example['width']])
      }
      
    else:
      gaze = [float(example['gaze_cx']) / 640, float(example['gaze_cy']) / 480]
      sp_gaze_points = [torch.FloatTensor(gaze)]
      sp_gaze_points_padded = torch.full((num_bbox, 2), 0).float()
      
      sp_gaze_points_padded[:len(sp_gaze_points), :] = torch.stack(
          sp_gaze_points
      )
      sp_gaze_points_padding = torch.full((num_bbox,), False)
      sp_gaze_points_padding[len(sp_gaze_points) :] = True

      target = {
        'gaze_point' : torch.stack([sp_gaze_points_padded]),
        'gaze_points_padding' : torch.stack([sp_gaze_points_padding]),
        'gaze_watch_outside' : torch.BoolTensor([False]).long(),
        "img_size": torch.FloatTensor([example['height'], example['width']])
      }
      # represents which objects have a gaze heatmap
      regression_padding = torch.full((num_queries, 1), True)
      regression_padding[: len(target["gaze_point"])] = False
      target['regression_padding'] = regression_padding
      
      gaze_points = torch.full((num_queries, num_bbox, 2), 0).float()
      gaze_points[: len(target['gaze_point']), :, :] = target["gaze_point"]
      target['gaze_point'] = gaze_points

      # gaze point padding
      gaze_points_padding = torch.full((num_queries, num_bbox), False)
      gaze_points_padding[: len(target["gaze_points_padding"]), :] = target[
          "gaze_points_padding"
      ]
      target["gaze_points_padding"] = gaze_points_padding

    img_size = target["img_size"].repeat(num_queries, 1)
    target["img_size"] = img_size
      
    # gaze watch outside
    gaze_watch_outside = torch.full((num_queries, 1), 0).float()
    gaze_watch_outside[: len(target["gaze_watch_outside"]), 0] = target["gaze_watch_outside"]
    target["gaze_watch_outside"] = gaze_watch_outside.long()

    gaze_heatmaps = []
    for gaze_points, gaze_point_padding, regression_padding in zip(
        target["gaze_point"], target['gaze_points_padding'], target["regression_padding"]
    ):
      if regression_padding:
        gaze_heatmap = torch.full(
          (gaze_heatmap_size, gaze_heatmap_size),
          float(1.0)
        )
      else:
        gaze_heatmap = []
        for (gaze_x, gaze_y), gaze_padding in zip(gaze_points, gaze_point_padding):
            if gaze_x == -1 or gaze_padding:
                continue 
            gaze_heatmap.append(get_label_map(
                torch.zeros((gaze_heatmap_size, gaze_heatmap_size)),
                [
                  gaze_x * gaze_heatmap_size,
                  gaze_y * gaze_heatmap_size,
                ],
                3,
                "Gaussian",
            ))
        gaze_heatmap = torch.stack(gaze_heatmap)
        gaze_heatmap = gaze_heatmap.sum(dim=0) / gaze_heatmap.sum(dim=0).max()
            
      gaze_heatmaps.append(gaze_heatmap)
    
    target["gaze_heatmaps"] = torch.stack(gaze_heatmaps)

    return target

# convert COCO format
def convert_obj_to_coco_format(example, data_args):
  if isinstance(example['labels'], str):
    ls_label = convert_str_to_list(example['labels'])
  if isinstance(example['bboxes'], str):
    ls_bbox = convert_str_to_list(example['bboxes'])

  nlabel, nbbox = [], []
  if example['gazeIdx'] == -1:
    # add head
    nlabel.append(ls_label[-1])
    nbbox.append(ls_bbox[-1])

    # add random objects
    topK = data_args.topknn - 1 # remove bbox head
    nlabel.extend(ls_label[:topK])
    nbbox.extend(ls_bbox[:topK])
  else:
    topK = data_args.topknn - 2 
    eudists = []
    for _, bbox in enumerate(ls_bbox[:-1]):
      gaze_point = (example['gaze_cx'], example['gaze_cy'])
      point_neibor = get_center_point(bbox)
      distance = eudis(gaze_point, point_neibor)
      eudists.append(distance)

    # list top items nearest gaze_target
    sensi_area = sensitive_area(eudists, topK) 

    # add head
    nlabel.append(ls_label[-1])
    nbbox.append(ls_bbox[-1])

    # add gaze object target
    nlabel.append(ls_label[example['gazeIdx']])
    nbbox.append(ls_bbox[example['gazeIdx']]) 

    # add nearest objects
    nlabel.extend([ls_label[i] for i in sensi_area])
    nbbox.extend([ls_bbox[i] for i in sensi_area])

  # convert to format coco
  area, bbox, category = [], [], []
  for _, (label, box) in enumerate(zip(nlabel, nbbox)):
    
    xmin, ymin, xmax, ymax = box
    
    o_width = xmax - xmin
    o_height = ymax - ymin

    area.append(o_width * o_height)
    bbox.append([xmin, ymin, o_width, o_height])
    category.append(label)

  return {
      'image_id' : example['seg'].split('.')[0],
      'objects' : {
          'area' : area,
          'bbox' : bbox,
          'category' : category
      },
      'gazeformer' : {
          'labels' : nlabel,
          'hx' : example['hx'],
          'hy' : example['hy'],
          'width' : example['width'],
          'height' : example['height'],
          'gaze_cx' : example['gaze_cx'],
          'gaze_cy' : example['gaze_cy'],
          'gazeIdx' : example['gazeIdx']
      }
  }

def get_center_point(bbox:List[float] = None):
  """Central bbox

  Args:
      bbox (List[float], optional): List of points bbox. Defaults to None.
      Examples: [x_min, y_min, x_max, y_max]

  Returns:
      Tuple: A central bbox
  """
  assert isinstance(bbox, list)
  x_point = (bbox[0] + bbox[2]) / 2
  y_point = (bbox[1] + bbox[3]) / 2
  
  return (x_point, y_point)

def eudis(point_target: List[float] = None, point_neibor: List[float] = None):
  """Calculate Euclidean distance between two points

  Args:
      point_target (List[float], optional): The anchor point. Defaults to None.
      point_neibor (List[float], optional): The neighbor point. Defaults to None.

  Returns:
      Float:
  """
  x_point = np.power((point_target[0] - point_neibor[0]), 2)
  y_point = np.power((point_target[1] - point_neibor[1]), 2)
  
  return np.sqrt(x_point + y_point)

def sensitive_area(eudists : List[float] = None, topK:int = 5):
  """Get sensitive area from anchor point. Ranking based on distance

  Args:
      eudists (List[float], optional): List of result euclidean distance. Defaults to None.
      topK (int, optional): Top K-nearest neighbor from anchor. Defaults to 5.

  Returns:
      _type_: _description_
  """
  sort_index = np.argsort(eudists)
  
  # skip index = 0 is target point
  return sort_index[1 : topK+1]

def get_angle_magnitude(p1, p2, dimension=2):
    # Add first dimension if it doesn't exist
    if len(p1.shape) == 1:
        p1 = p1.unsqueeze(0)
    if len(p2.shape) == 1:
        p2 = p2.unsqueeze(0)

    vx, vy = p2[:, 0] - p1[:, 0], p2[:, 1] - p1[:, 1]
    
    magnitude = torch.sqrt(vx**2 + vy**2)
    # phi range is [-pi, pi]
    phi = torch.atan2(vy, vx)

    # Scale magnitude to be between 0 and 1
    magnitude = magnitude / math.sqrt(2)

    # Scale phi to be between 0 and 1
    phi = (phi + torch.pi) / (2 * torch.pi)

    return torch.cat([phi.unsqueeze(1), magnitude.unsqueeze(1)], dim=1)
