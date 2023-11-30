import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, vgg19, VGG19_Weights

# Global Feature Extractor
class GlobalFeatureExtractor(nn.Module):
    """
    This class implements the global feature extractor.
    Utilizes the ResNet50 architecture.
    """
    def __init__(self):
        super(GlobalFeatureExtractor, self).__init__()
        # using the pretrained ResNet50 model
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # remove the last two layers
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

    def forward(self, x):
        x = self.resnet(x)
        print(f"Global Feature Shape: {x.shape}")
        return x
    
# Local Feature Extractor
class LocalFeatureExtractor(nn.Module):
    def __init__(self):
        super(LocalFeatureExtractor, self).__init__()
        # Use VGG19 up to a certain layer
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT)
        self.vgg = nn.Sequential(*list(self.vgg.features)[:21])  # Up to relu4_1

        # Additional layers to match the channel size
        self.extra_layers = nn.Sequential(
            nn.Conv2d(512, 2048, kernel_size=1),  # Match the channel size to 2048
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))          # Adjust size to 7x7
        )

    def forward(self, x):
        x = self.vgg(x)
        x = self.extra_layers(x)
        print(f"Local Feature Shape: {x.shape}")
        return x
