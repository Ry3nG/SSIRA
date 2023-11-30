import logging
from torch import nn
from .feature_extractor import GlobalFeatureExtractor, LocalFeatureExtractor
from .adaptive_feature_integration import AdaptiveFeatureIntegration
from .attention import ChannelAttentionModule, SpatialAttentionModule
from .self_supervised_learning import DegradationLayers, ReconstructionHead


class GLINTnet(nn.Module):
    def __init__(self, num_classes, is_classification=True):
        super(GLINTnet, self).__init__()
        self.global_feature_extractor = GlobalFeatureExtractor()
        self.local_feature_extractor = LocalFeatureExtractor()
        self.adaptive_feature_integration = AdaptiveFeatureIntegration()
        self.channel_attention = ChannelAttentionModule(in_channels=2048)
        self.spatial_attention = SpatialAttentionModule()

        # define classification/regression head
        if is_classification:
            self.head = nn.Sequential(
                nn.Linear(
                    100352, 1024  # Updated input size
                ),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, num_classes),
                nn.Softmax(dim=1) if num_classes > 2 else nn.Sigmoid(),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(
                    100352, 1024  # Updated input size
                ),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 1),
            )

    def forward(self, x):
        print(f"Input shape in GLINet forward pass: {x.shape}")
        global_features = self.global_feature_extractor(x)  # (batch_size, 2048, 7, 7)
        local_features = self.local_feature_extractor(x)

        # Print shapes for debugging
        print("Global features shape in GLINet forward pass:", global_features.shape)
        print("Local features shape in GLINet forward pass:", local_features.shape)

        # Apply attention modules
        global_features = self.channel_attention(global_features)
        global_features = self.spatial_attention(global_features)
        local_features = self.channel_attention(local_features)
        local_features = self.spatial_attention(local_features)

        integrated_features = self.adaptive_feature_integration(
            global_features, local_features
        )

        # Print shape for debugging
        print(
            "Integrated features shape in GLINet forward pass:",
            integrated_features.shape,
        )

        # Flatten the integrated features
        integrated_features_flattened = integrated_features.view(
            integrated_features.size(0), -1
        )
        # Adjust the number of input features in the first Linear layer if necessary
        print(
            "Flattened features shape in GLINet forward pass:",
            integrated_features_flattened.shape,
        )
        output = self.head(integrated_features_flattened)
        return output


class GLINTnetSelfSupervised(nn.Module):
    def __init__(self, base_model, input_features, manipulation_options):
        super(GLINTnetSelfSupervised, self).__init__()
        print("GLINTnetSS initialized")
        print(f"Input features: {input_features}")
        print(f"Manipulation options: {manipulation_options}")
        self.base_model = base_model
        self.degradation_layer = DegradationLayers(manipulation_options)
        self.reconstruction_head = ReconstructionHead(input_features)

    def forward(self, x):
        print(f"Input shape in GLINetSS forward pass: {x.shape}")
        degraded_x = self.degradation_layer(x)
        print(f"Degraded shape in GLINetSS forward pass: {degraded_x.shape}")

        features = self.base_model(degraded_x) # base model is GLINTnet
        print(f"Features shape in GLINetSS forward pass: {features.shape}")

        reconstructed_x = self.reconstruction_head(features)
        print(f"Reconstructed shape in GLINetSS forward pass: {reconstructed_x.shape}")

        # Ensure that reconstructed_x is on the same device as input
        reconstructed_x = reconstructed_x.to(x.device)

        return reconstructed_x, degraded_x
