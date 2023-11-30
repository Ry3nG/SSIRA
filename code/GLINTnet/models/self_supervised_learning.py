import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 as cv
import random
from utils.transforms import image_manipulation

class DegradationLayers(nn.Module):
    def __init__(self, manipulation_options):
        super(DegradationLayers, self).__init__()
        self.manipulation_options = manipulation_options

    def forward(self, x_batch):
        degraded_images = []

        for i in range(x_batch.size(0)):  # Iterate over batch
            x = x_batch[i]  # Extract single image tensor

            # Ensure it's detached and on CPU
            x_cv = x.detach().cpu().numpy().transpose(1, 2, 0)
            x_cv = (x_cv * 255).astype(np.uint8)
            x_cv = cv.cvtColor(x_cv, cv.COLOR_RGB2BGR)

            opt = random.choice(self.manipulation_options)
            degraded_img_cv = image_manipulation(x_cv, opt)

            degraded_img_cv = cv.cvtColor(degraded_img_cv, cv.COLOR_BGR2RGB)
            degraded_img_tensor = torch.from_numpy(degraded_img_cv).permute(2, 0, 1).float() / 255.0

            degraded_images.append(degraded_img_tensor.unsqueeze(0))

        degraded_images_tensor = torch.cat(degraded_images, dim=0)
        print(f"Degraded images tensor shape: {degraded_images_tensor.shape}")
        return degraded_images_tensor.to(x_batch.device)  # Move to the same device as input


class ReconstructionHead(nn.Module):
    def __init__(self, input_features):
        super(ReconstructionHead, self).__init__()

        # first convTranspose2d layer should accept input_features input channels
        self.upscaling_layers = nn.Sequential(
            nn.ConvTranspose2d(input_features, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024),

            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),

            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh() 
        )

    def forward(self, x):
        x = self.upscaling_layers(x)
        return x
