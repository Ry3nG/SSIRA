import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconstructionLoss(nn.Module):
    """
    Reconstruction Loss for the self-supervised learning phase.
    This can be a simple Mean Squared Error (MSE) loss
    comparing the reconstructed image to the original image.
    """
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, reconstructed, original):
        return F.mse_loss(reconstructed, original)


class AestheticScoreLoss(nn.Module):
    """
    Aesthetic Score Loss for the supervised learning phase.
    This can be an L1 loss (Mean Absolute Error) or L2 loss (MSE),
    depending on how the aesthetic scores are distributed.
    """
    def __init__(self):
        super(AestheticScoreLoss, self).__init__()

    def forward(self, predicted_scores, true_scores):
        # Ensure that both predicted_scores and true_scores have the same shape
        if predicted_scores.dim() > 1:  # Check if predicted_scores is not already flattened
            predicted_scores = predicted_scores.view(-1)  # Flatten the predicted_scores

        return F.l1_loss(predicted_scores, true_scores)
