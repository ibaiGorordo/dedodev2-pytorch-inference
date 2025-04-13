import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dedodev2.utils import sample_keypoints, to_pixel_coords, to_normalized_coords, get_best_device



class DeDoDeDetector(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 *args,
                 remove_borders = False,
                 H=784,
                 W=784,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        import torchvision.transforms as transforms
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.remove_borders = remove_borders
        self.H = H
        self.W = W
        
    def forward(self, input_tensor):
        features, sizes = self.encoder(input_tensor)
        logits = 0
        context = None
        scales = ["8", "4", "2", "1"]
        for idx, (feature_map, scale) in enumerate(zip(reversed(features), scales)):
            delta_logits, context = self.decoder(feature_map, context = context, scale = scale)
            logits = logits + delta_logits.float() # ensure float (need bf16 doesnt have f.interpolate)
            if idx < len(scales) - 1:
                size = sizes[-(idx+2)]
                logits = F.interpolate(logits, size = size, mode = "bicubic", align_corners = False)
                context = F.interpolate(context.float(), size = size, mode = "bilinear", align_corners = False)
        return {"keypoint_logits" : logits.float()}
    
    @torch.inference_mode()
    def detect(self, img, num_keypoints = 10_000):
        input_tensor = self.preprocess(img)
        keypoint_logits = self.forward(input_tensor)["keypoint_logits"]
        B,K,H,W = keypoint_logits.shape
        keypoint_p = keypoint_logits.reshape(B, K*H*W).softmax(dim=-1).reshape(B, K, H*W).sum(dim=1)
        keypoints, confidence = sample_keypoints(keypoint_p.reshape(B,H,W), 
                                  use_nms = False, sample_topk = True, num_samples = num_keypoints, 
                                  return_scoremap=True, sharpen = False, upsample = False,
                                  increase_coverage=True, remove_borders = self.remove_borders)
        return {"keypoints": keypoints, "confidence": confidence}

    def preprocess(self, img, device=get_best_device()):
        input_img = cv2.resize(img, (self.W, self.H))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        standard_im = input_img/255.
        return self.normalizer(torch.from_numpy(standard_im).permute(2,0,1)).float().to(device)[None]

    def to_pixel_coords(self, x, H, W):
        return to_pixel_coords(x, H, W)
    
    def to_normalized_coords(self, x, H, W):
        return to_normalized_coords(x, H, W)