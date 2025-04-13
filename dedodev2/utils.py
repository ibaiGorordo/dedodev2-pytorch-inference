import torch.nn.functional as F
import torch
import numpy as np
import cv2
from cmap import Colormap


cmap = Colormap('matlab:hsv')


def get_best_device(verbose = False):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    if verbose: print (f"Fastest device found is: {device}")
    return device


def get_grid(B,H,W, device = get_best_device()):
    x1_n = torch.meshgrid(
        *[
            torch.linspace(
                -1 + 1 / n, 1 - 1 / n, n, device=device
            )
            for n in (B, H, W)
        ],
        indexing='ij'
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)
    return x1_n


@torch.no_grad()
def sample_keypoints(scoremap, num_samples = 8192, device = get_best_device(), use_nms = True, 
                     sample_topk = False, return_scoremap = False, sharpen = False, upsample = False,
                     increase_coverage = False, remove_borders = False):
    #scoremap = scoremap**2
    log_scoremap = (scoremap+1e-10).log()
    if upsample:
        log_scoremap = F.interpolate(log_scoremap[:,None], scale_factor = 3, mode = "bicubic", align_corners = False)[:,0]#.clamp(min = 0)
        scoremap = log_scoremap.exp()
    B,H,W = scoremap.shape
    if increase_coverage:
        weights = (-torch.linspace(-2, 2, steps = 51, device = device)**2).exp()[None,None]
        # 10000 is just some number for maybe numerical stability, who knows. :), result is invariant anyway
        local_density_x = F.conv2d((scoremap[:,None]+1e-6)*10000,weights[...,None,:], padding = (0,51//2))
        local_density = F.conv2d(local_density_x, weights[...,None], padding = (51//2,0))[:,0]
        scoremap = scoremap * (local_density+1e-8)**(-1/2)
    grid = get_grid(B,H,W, device=device).reshape(B,H*W,2)
    if sharpen:
        laplace_operator = torch.tensor([[[[0,1,0],[1,-4,1],[0,1,0]]]], device = device)/4
        scoremap = scoremap[:,None] - 0.5 * F.conv2d(scoremap[:,None], weight = laplace_operator, padding = 1)
        scoremap = scoremap[:,0].clamp(min = 0)
    if use_nms:
        scoremap = scoremap * (scoremap == F.max_pool2d(scoremap, (3, 3), stride = 1, padding = 1))
    if remove_borders:
        frame = torch.zeros_like(scoremap)
        # we hardcode 4px, could do it nicer, but whatever
        frame[...,4:-4, 4:-4] = 1
        scoremap = scoremap * frame
    if sample_topk:
        inds = torch.topk(scoremap.reshape(B,H*W), k = num_samples).indices
    else:
        inds = torch.multinomial(scoremap.reshape(B,H*W), num_samples = num_samples, replacement=False)
    kps = torch.gather(grid, dim = 1, index = inds[...,None].expand(B,num_samples,2))
    if return_scoremap:
        return kps, torch.gather(scoremap.reshape(B,H*W), dim = 1, index = inds)
    return kps

imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
imagenet_std = torch.tensor([0.229, 0.224, 0.225])


def to_pixel_coords(flow, h1, w1):
    flow = (
        torch.stack(
            (
                w1 * (flow[..., 0] + 1) / 2,
                h1 * (flow[..., 1] + 1) / 2,
            ),
            axis=-1,
        )
    )
    return flow

def to_normalized_coords(flow, h1, w1):
    flow = (
        torch.stack(
            (
                2 * (flow[..., 0]) / w1 - 1,
                2 * (flow[..., 1]) / h1 - 1,
            ),
            axis=-1,
        )
    )
    return flow

def dual_softmax_matcher(desc_A: tuple['B','N','C'], desc_B: tuple['B','M','C'], inv_temperature = 1, normalize = False):
    if len(desc_A.shape) < 3:
        desc_A, desc_B = desc_A[None], desc_B[None]
    B, N, C = desc_A.shape
    if normalize:
        desc_A = desc_A/desc_A.norm(dim=-1,keepdim=True)
        desc_B = desc_B/desc_B.norm(dim=-1,keepdim=True)
        corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    else:
        corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    P = corr.softmax(dim = -2) * corr.softmax(dim= -1)
    return P



def draw_kpts(im, kpts):
    max_val = im.shape[0] * im.shape[1]
    radius = max(int(3 / 1000 * max(im.shape[0], im.shape[1])), 2)
    for x, y in kpts.cpu().numpy():
        x = int(x)
        y = int(y)
        color = np.array(cmap(x * y / max_val))[2::-1] * 255
        cv2.circle(im, (x, y), radius, color, -1)
    return im

def draw_matches(im_1, kpts_1, im_2, kpts_2):
    kpts_1 = kpts_1.cpu().numpy()
    kpts_2 = kpts_2.cpu().numpy()

    im_1, im_2 = np.array(im_1), np.array(im_2)
    radius_1 = max(int(3 / 1000 * max(im_1.shape[0], im_1.shape[1])), 2)
    radius_2 = max(int(3 / 1000 * max(im_2.shape[0], im_2.shape[1])), 2)
    line_width = max(int(2 / 1000 * max(im_1.shape[0], im_1.shape[1])), 1)

    max_val = im_1.shape[0] * im_1.shape[1]

    combined = np.hstack((im_1, im_2))
    for x1, y1, x2, y2 in zip(kpts_1[:, 0], kpts_1[:, 1], kpts_2[:, 0], kpts_2[:, 1]):

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2) + im_1.shape[1]
        y2 = int(y2)

        color = np.array(cmap(x1 * y1 / max_val))[2::-1] * 255

        cv2.circle(combined, (x1, y1), radius_1, color, -1)
        cv2.circle(combined, (x2, y2), radius_2, color, -1)
        cv2.line(combined, (x1, y1), (x2, y2), color, line_width)

    return combined