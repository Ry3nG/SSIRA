import torch
import torch.nn as nn
import torch.nn.functional as F

class DegradationCNN(nn.Module):
    def __init__(self, num_degradation_types):
        super(DegradationCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, num_degradation_types)
        self.fc3 = nn.Linear(512, 1)  # For level of degradation

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 64 * 64)
        x = F.relu(self.fc1(x))
        
        # Output for degradation type
        type_output = self.fc2(x)
        
        # Output for degradation level (assuming it's a continuous value)
        level_output = self.fc3(x)
        
        return type_output, level_output

# Example usage
# model = DegradationCNN(num_degradation_types=5) # Assuming 5 types of degradation
