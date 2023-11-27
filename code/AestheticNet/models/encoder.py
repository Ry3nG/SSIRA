import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Load a pre-trained ResNet50 model
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Remove the fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

    def forward(self, x):
        x = self.resnet(x) 
        return x
