
import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5,device = None):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, features1, features2):
        # Normalize the features
        features1 = nn.functional.normalize(features1, dim=1)
        features2 = nn.functional.normalize(features2, dim=1)
        
        # Compute the cosine similarity
        similarity_matrix = torch.mm(features1, features2.T)
        
        # Create labels for the positive pairs (diagonal elements of the matrix)
        labels = torch.arange(features1.size(0)).to(self.device)
        
        # Calculate the contrastive loss
        loss = nn.CrossEntropyLoss()(similarity_matrix / self.temperature, labels)
        return loss