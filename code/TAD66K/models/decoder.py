import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, input_channels=1280, output_channels=3):
        super(Decoder, self).__init__()
        
        # Upscaling and convolutional layers to reconstruct the image
        self.upscaling_layers = nn.Sequential(
            nn.ConvTranspose2d(input_channels, 640, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(640),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(640, 320, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(320, 160, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(160, 80, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(80, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # Using Tanh for [-1, 1] range; change to Sigmoid for [0, 1]
        )

    def forward(self, x):
        x = x.view(x.size(0), 1280, 1, 1)  # Reshape input to match the expected size
        x = self.upscaling_layers(x)
        return x
