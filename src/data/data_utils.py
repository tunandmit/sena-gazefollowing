import torch
import numpy as np


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor

def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray

def get_label_map(img, pt, sigma, pdf="Gaussian"):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [
        pt[0].round().int().item() - 3 * sigma,
        pt[1].round().int().item() - 3 * sigma,
    ]
    br = [
        pt[0].round().int().item() + 3 * sigma + 1,
        pt[1].round().int().item() + 3 * sigma + 1,
    ]
    if ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0:
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if pdf == "Gaussian":
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    elif pdf == "Cauchy":
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma**2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0] : img_y[1], img_x[0] : img_x[1]] += g[g_y[0] : g_y[1], g_x[0] : g_x[1]]
    img = img / np.max(img)  # normalize heatmap so it has max value of 1

    return to_torch(img)


def get_gaze_cone(
    head_center_point,
    gaze_vector,
    out_size=(64, 64),
    cone_angle=120,
):
    width, height = out_size
    eye_coords = (
        (
            head_center_point
            * torch.tensor([width, height], device=head_center_point.device)
        )
        .unsqueeze(1)
        .unsqueeze(1)
    )
    gaze_coords = (
        (
            (head_center_point + gaze_vector)
            * torch.tensor([width, height], device=head_center_point.device)
        )
        .unsqueeze(1)
        .unsqueeze(1)
    )

    pixel_mat = (
        torch.stack(
            torch.meshgrid(
                [torch.arange(1, width + 1), torch.arange(1, height + 1)]
            ),
            dim=-1,
        )
        .unsqueeze(0)
        .repeat(head_center_point.shape[0], 1, 1, 1)
        .to(head_center_point.device)
    )

    dot_prod = torch.sum((pixel_mat - eye_coords) * (gaze_coords - eye_coords), dim=-1)
    gaze_vector_norm = torch.sqrt(torch.sum((gaze_coords - eye_coords) ** 2, dim=-1))
    pixel_mat_norm = torch.sqrt(torch.sum((pixel_mat - eye_coords) ** 2, dim=-1))

    theta = cone_angle * (torch.pi / 180)
    beta = torch.acos(dot_prod / (gaze_vector_norm * pixel_mat_norm))

    # Create mask where true if beta is less than theta/2
    pixel_mat_presence = beta < (theta / 2)

    gaze_cones = dot_prod / (gaze_vector_norm * pixel_mat_norm)

    # Zero out values outside the gaze cone
    gaze_cones[~pixel_mat_presence] = 0
    gaze_cones = torch.clamp(gaze_cones, 0, None)

    return gaze_cones