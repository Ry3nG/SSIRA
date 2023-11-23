import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Use ResNet18 as backbone with updated weights parameter
        self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features, 128
        )  # Feature embedding

    def forward(self, input1, input2):
        output1 = self.dropout(self.backbone(input1))
        output2 = self.dropout(self.backbone(input2))
        return output1, output2

    def get_embedding(self, input):
        return self.backbone(input)
