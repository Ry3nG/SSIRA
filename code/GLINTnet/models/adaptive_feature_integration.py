import torch
import torch.nn as nn

def calc_mean_std(features):
    """
    Calculates the mean and standard deviation of the features.
    """
    batch_size, channels = features.size()[:2]
    features_mean = features.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    features_std = features.view(batch_size, channels, -1).std(dim=2).view(batch_size, channels, 1, 1)

    return features_mean, features_std

class AdaptiveFeatureIntegration(nn.Module):
    """
    Adaptive Feature Integration (AFI) inspired by AdaIN.
    This module aligns the mean and standard deviation of the local features
    to those of the global features.
    """
    def __init__(self):
        super(AdaptiveFeatureIntegration, self).__init__()

    def forward(self, global_features, local_features):
        global_mean, global_std = calc_mean_std(global_features)
        local_mean, local_std = calc_mean_std(local_features)

        normalized_local_features = global_std*(local_features - local_mean)/local_std + global_mean

        return normalized_local_features