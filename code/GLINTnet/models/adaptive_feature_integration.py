import torch
import torch.nn as nn

def calc_mean_std(features):
    batch_size, channels = features.size()[:2]
    reshaped_features = features.view(batch_size, channels, -1)
    
    #print(f"Reshaped Features - Min: {reshaped_features.min()}, Max: {reshaped_features.max()}, Mean: {reshaped_features.mean()}, Std: {reshaped_features.std()}")

    features_mean = reshaped_features.mean(dim=2).view(batch_size, channels, 1, 1)
    features_std = reshaped_features.std(dim=2).view(batch_size, channels, 1, 1)+ 1e-6

    #print(f"Features Mean - Min: {features_mean.min()}, Max: {features_mean.max()}, Mean: {features_mean.mean()}, Std: {features_mean.std()}")
    #print(f"Features Std - Min: {features_std.min()}, Max: {features_std.max()}, Mean: {features_std.mean()}, Std: {features_std.std()}")

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

        """
        print(f"Global mean - Min: {global_mean.min()}, Max: {global_mean.max()}, Mean: {global_mean.mean()}, Std: {global_mean.std()}")
        print(f"Global std - Min: {global_std.min()}, Max: {global_std.max()}, Mean: {global_std.mean()}, Std: {global_std.std()}")
        print(f"Local mean - Min: {local_mean.min()}, Max: {local_mean.max()}, Mean: {local_mean.mean()}, Std: {local_mean.std()}")
        print(f"Local std - Min: {local_std.min()}, Max: {local_std.max()}, Mean: {local_std.mean()}, Std: {local_std.std()}")
        """

        normalized_local_features = global_std * (local_features - local_mean) / (local_std) + global_mean

        """
        print(f"Normalized local features shape: {normalized_local_features.shape}")
        print(f"Normalized local features min: {normalized_local_features.min()}")
        print(f"Normalized local features max: {normalized_local_features.max()}")
        print(f"Normalized local features mean: {normalized_local_features.mean()}")
        print(f"Normalized local features std: {normalized_local_features.std()}")
        """

        return normalized_local_features