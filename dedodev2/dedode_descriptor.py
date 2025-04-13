import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dedodev2.utils import get_best_device, to_normalized_coords


class DeDoDeDescriptor(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 *args,
                 H=784,
                 W=784,
                 device=get_best_device(),
                 **kwargs,
                 ) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        import torchvision.transforms as transforms
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.H = H
        self.W = W
        self.device = device
        
    def forward(self, input_tensor):
        features, sizes = self.encoder(input_tensor)
        descriptor = 0
        context = None
        scales = self.decoder.scales
        for idx, (feature_map, scale) in enumerate(zip(reversed(features), scales)):
            delta_descriptor, context = self.decoder(feature_map, scale = scale, context = context)
            descriptor = descriptor + delta_descriptor
            if idx < len(scales) - 1:
                size = sizes[-(idx+2)]
                descriptor = F.interpolate(descriptor, size = size, mode = "bilinear", align_corners = False)
                context = F.interpolate(context, size = size, mode = "bilinear", align_corners = False)
        return descriptor
    
    @torch.inference_mode()
    def describe_keypoints(self, img, keypoints):
        print(keypoints.shape)
        input_tensor = self.preprocess(img)
        description_grid = self.forward(input_tensor)
        kpts = to_normalized_coords(keypoints[None], img.shape[0], img.shape[1])
        described_keypoints = F.grid_sample(description_grid.float(), kpts[:,None], mode = "bilinear", align_corners = False)[:,:,0].mT
        return described_keypoints
    
    def preprocess(self, img):

        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (self.W, self.H))
        input_img = input_img.astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(input_img).permute(2, 0, 1)
        input_tensor = self.normalizer(input_tensor)
        input_tensor = input_tensor.float().to(self.device)[None]

        return input_tensor
