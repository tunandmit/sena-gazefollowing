import torch
import numpy as np
from ..data.data_utils import to_numpy



def get_multi_hot_map(gaze_pts, out_res, device=torch.device("cuda")):
    h, w = out_res
    target_map = torch.zeros((h, w), device=device).long()
    for p in gaze_pts:
        if p[0] >= 0:
            x, y = map(int, [p[0] * float(w), p[1] * float(h)])
            x = min(x, w - 1)
            y = min(y, h - 1)
            target_map[y, x] = 1

    return target_map


def get_heatmap_peak_coords(heatmap):
    np_heatmap = to_numpy(heatmap)
    idx = np.unravel_index(np_heatmap.argmax(), np_heatmap.shape)
    pred_y, pred_x = map(float, idx)

    return pred_x, pred_y


def get_l2_dist(p1, p2):
    return torch.sqrt((p1[:, 0] - p2[:, 0]) ** 2 + (p1[:, 1] - p2[:, 1]) ** 2)