import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.features = mobilenet.features

        # Replace the classifier part with an adaptive average pool to handle variable input sizes
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        return x
