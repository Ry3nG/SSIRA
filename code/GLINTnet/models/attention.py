import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttentionModule(nn.Module):
    """
    implements channel attention,
    focusing on 'what' is meaningful given an input feature map.
    """
    def __init__(self, in_channels, reduction_ratio = 16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_pooled = self.avg_pool(x)
        avg_pooled = self.fc(avg_pooled.view(avg_pooled.size(0), -1))
        
        max_pooled = self.max_pool(x)
        max_pooled = self.fc(max_pooled.view(max_pooled.size(0), -1))
        print("avg_pooled.shape from channel attention: ", avg_pooled.shape)
        print("max_pooled.shape from channel attention: ", max_pooled.shape)

        return x * (avg_pooled.view(avg_pooled.size(0), -1, 1, 1) + max_pooled.view(max_pooled.size(0), -1, 1, 1))
    
class SpatialAttentionModule(nn.Module):
    """
    Implements spatial attention, focusing on 'where' is an informative part in the feature map.
    """
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pooled = torch.mean(x, dim=1, keepdim=True)
        max_pooled, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_pooled, max_pooled], dim=1)
        attention = self.sigmoid(self.conv(pooled))

        print("attention.shape from spatial attention: ", attention.shape)
        print("x.shape from spatial attention: ", x.shape)
        
        return x * attention