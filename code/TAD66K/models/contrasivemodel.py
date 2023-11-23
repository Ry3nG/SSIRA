import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ContrastiveModel(nn.Module):
    def __init__(self):
        super(ContrastiveModel, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()  # Remove the final fully connected layer

    def forward(self, x):
        # Assuming x is a tuple of (degraded_images, original_images)
        degraded_images, original_images = x
        features_degraded = self.backbone(degraded_images)
        features_original = self.backbone(original_images)
        return features_degraded, features_original
