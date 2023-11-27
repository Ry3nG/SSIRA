import torch.nn as nn
class Decoder(nn.Module):
    def __init__(self, input_channels=2048, output_channels=3):
        super(Decoder, self).__init__()
        
        # Adjust the first ConvTranspose2d layer to accept 2048 input channels
        self.upscaling_layers = nn.Sequential(
            nn.ConvTranspose2d(input_channels, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024),
            
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),

            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # Adjust this based on your input normalization
        )

    def forward(self, x):
        x = self.upscaling_layers(x)
        return x
