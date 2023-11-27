import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class AestheticNet(nn.Module):
    def __init__(self, encoded_image_size=1280, num_classes=1):
        super(AestheticNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(input_channels=encoded_image_size)

        # Regression head for predicting aesthetic scores
        self.regression_head = nn.Sequential(
            nn.Linear(encoded_image_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Sigmoid()  # Use Sigmoid if the scores are normalized between 0 and 1
        )

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
            return aesthetic_score
