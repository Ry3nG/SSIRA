import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class AestheticNet(nn.Module):
    def __init__(self, encoded_image_size=2048, num_classes=1,input_image_size=224):
        super(AestheticNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(input_channels=encoded_image_size)

        # Compute the flattened size of the encoded image
        self.flattened_size = self._get_flattened_size(input_image_size)
        # Regression head for predicting aesthetic scores
        self.regression_head = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def _get_flattened_size(self, input_image_size):
        # Assuming square input images and standard ResNet output size
        feature_map_size = input_image_size // 32  # ResNet reduces size by a factor of 32
        return feature_map_size * feature_map_size * 2048
    
    def forward(self, x, phase='pretext'):
        if phase == 'pretext':
            # In the pretext phase, use both encoder and decoder for reconstruction
            encoded_features = self.encoder(x)
            reconstructed_image = self.decoder(encoded_features)
            return reconstructed_image
        elif phase == 'aesthetic':
            # In the aesthetic assessment phase, use only the encoder and regression head
            encoded_features = self.encoder(x)
            encoded_features = torch.flatten(encoded_features, 1)  # Flatten the features
            aesthetic_score = self.regression_head(encoded_features)
            #print(aesthetic_score)
            return aesthetic_score
