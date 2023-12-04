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
        #print(f"Input shape in GLINet forward pass: {x.shape}")
        global_features = self.global_feature_extractor(x)  # (batch_size, 2048, 7, 7)
        local_features = self.local_feature_extractor(x)

        # Print shapes for debugging
        #print("Global features shape in GLINet forward pass:", global_features.shape)
        #print("Local features shape in GLINet forward pass:", local_features.shape)

        # Apply attention modules
        global_features = self.channel_attention(global_features)
        global_features = self.spatial_attention(global_features)
        local_features = self.channel_attention(local_features)
        local_features = self.spatial_attention(local_features)


        integrated_features = self.adaptive_feature_integration(
            global_features, local_features
        )

        """
        # Print shape for debugging
        print(
            "Integrated features shape in GLINet forward pass:",
            integrated_features.shape,
        )
        """

        # Flatten the integrated features
        integrated_features_flattened = integrated_features.view(
            integrated_features.size(0), -1
        )
        """
        
        # Adjust the number of input features in the first Linear layer if necessary
        print(
            "Flattened features shape in GLINet forward pass:",
            integrated_features_flattened.shape,
        )
        """
        output = self.head(integrated_features_flattened)
        return output


class GLINTnetSelfSupervised(nn.Module):
    def __init__(self, base_model, input_features, manipulation_options):
        super(GLINTnetSelfSupervised, self).__init__()
        print("GLINTnetSS initialized")
        #print(f"Input features: {input_features}")
        #print(f"Manipulation options: {manipulation_options}")
        self.base_model = base_model
        self.degradation_layer = DegradationLayers(manipulation_options)
        self.reconstruction_head = ReconstructionHead(input_features)

    def forward(self, x):
        degraded_x = self.degradation_layer(x)
        #print(f"Degraded x shape in GLINTnetSS forward pass: {degraded_x.shape}")

        # Extract features directly before the classification/regression head
        global_features = self.base_model.global_feature_extractor(degraded_x)
        local_features = self.base_model.local_feature_extractor(degraded_x)
        integrated_features = self.base_model.adaptive_feature_integration(global_features, local_features)
        #print(f"Integrated features shape in GLINTnetSS forward pass: {integrated_features.shape}")

        
        
        # Optionally, apply attention mechanisms if needed
        integrated_features = self.base_model.channel_attention(integrated_features)
        #print(f"Integrated features shape after channel attention in GLINTnetSS forward pass: {integrated_features.shape}")
        #print(f"Integrated features min after channel attention in GLINTnetSS forward pass: min: {integrated_features.min()}, max: {integrated_features.max()}, mean: {integrated_features.mean()}, std: {integrated_features.std()}")
        integrated_features = self.base_model.spatial_attention(integrated_features)
        #print(f"Integrated features shape after spatial attention in GLINTnetSS forward pass: {integrated_features.shape}")
        #print(f"Integrated features min after spatial attention in GLINTnetSS forward pass: min: {integrated_features.min()}, max: {integrated_features.max()}, mean: {integrated_features.mean()}, std: {integrated_features.std()}")

        #print(f"Input to ReconstructionHead shape: {integrated_features.shape}")
        #print(f"Input to ReconstructionHead - Min: {integrated_features.min()}, Max: {integrated_features.max()}, Mean: {integrated_features.mean()}, Std: {integrated_features.std()}")
        
        reconstructed_x = self.reconstruction_head(integrated_features)
        #print(f"Reconstructed x shape in GLINTnetSS forward pass: {reconstructed_x.shape}")
        #print(f"Reconstructed x min in GLINTnetSS forward pass: min: {reconstructed_x.min()}, max: {reconstructed_x.max()}, mean: {reconstructed_x.mean()}, std: {reconstructed_x.std()}")
        return reconstructed_x, degraded_x
