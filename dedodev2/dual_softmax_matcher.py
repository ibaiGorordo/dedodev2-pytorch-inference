import torch
import torch.nn as nn
from dedodev2.utils import dual_softmax_matcher, to_pixel_coords, to_normalized_coords

class DualSoftMaxMatcher(nn.Module):
    @torch.inference_mode()
    def match(self, keypoints_A, descriptions_A,
              keypoints_B, descriptions_B,
              normalize = False, inv_temp = 1, threshold = 0.0):
        P = dual_softmax_matcher(descriptions_A, descriptions_B,
                                 normalize = normalize, inv_temperature=inv_temp,
                                 )
        inds = torch.nonzero((P[0] == P[0].max(dim=-1, keepdim=True).values)
                             * (P[0] == P[0].max(dim=-2, keepdim=True).values) * (P[0] > threshold))
        matches_A = keypoints_A[0, inds[:, 0]]
        matches_B = keypoints_B[0, inds[:, 1]]
        P = P[0, inds[:, 0], inds[:, 1]]
        return inds[:, 0], inds[:, 1], P

    def to_pixel_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_pixel_coords(x_A, H_A, W_A), to_pixel_coords(x_B, H_B, W_B)
    
    def to_normalized_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_normalized_coords(x_A, H_A, W_A), to_normalized_coords(x_B, H_B, W_B)