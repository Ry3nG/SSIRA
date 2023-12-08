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
        """
        print(f"Degraded images tensor shape: {degraded_images_tensor.shape}")
        print(f"Degraded images tensor min: {degraded_images_tensor.min()}")
        print(f"Degraded images tensor max: {degraded_images_tensor.max()}")
        print(f"Degraded images tensor mean: {degraded_images_tensor.mean()}")
        print(f"Degraded images tensor std: {degraded_images_tensor.std()}")
        """

        return degraded_images_tensor.to(x_batch.device)  # Move to the same device as input

class ReconstructionHead(nn.Module):
    def __init__(self, input_features):
        super(ReconstructionHead, self).__init__()
        print(f"Input features: {input_features}")

        self.conv1 = nn.ConvTranspose2d(input_features, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(512)

        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(64)

        self.final_conv = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final_act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        #self.print_stats(x, "conv1")
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        #self.print_stats(x, "conv2")
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        #self.print_stats(x, "conv3")
        x = self.relu3(x)
        x = self.conv4(x)
        #self.print_stats(x, "conv4")
        x = self.relu4(x)
        x = self.bn4(x)
        x = self.final_conv(x)
        #self.print_stats(x, "final_conv")
        x = self.final_act(x)
        #self.print_stats(x, "final_output")
        return x

    def print_stats(self, tensor, label):
        print(f"{label} shape: {tensor.shape}")
        print(f"{label} min: {tensor.min().item()}")
        print(f"{label} max: {tensor.max().item()}")
        print(f"{label} mean: {tensor.mean().item()}")
        print(f"{label} std: {tensor.std().item()}")