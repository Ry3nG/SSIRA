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
    # Aesthetic Score Loss for the supervised learning phase.
    # This can be an L1 loss (Mean Absolute Error) or L2 loss (MSE),
    # depending on how the aesthetic scores are distributed.

    def __init__(self):
        super(AestheticScoreLoss, self).__init__()

    def forward(self, predicted_scores, true_scores):
        # Ensure that both predicted_scores and true_scores have the same shape
        if predicted_scores.dim() > 1:
            predicted_scores = predicted_scores.view(-1)  # Flatten the predicted_scores

        # Convert both tensors to float type
        predicted_scores = predicted_scores.float()
        true_scores = true_scores.float()

        return F.mse_loss(predicted_scores, true_scores)


class EMDLoss(nn.Module):
    def __init__(self, dist_r=2):
        super(EMDLoss, self).__init__()
        self.dist_r = dist_r

    def forward(self, p_target, p_estimate):
        assert p_target.shape == p_estimate.shape
        cdf_target = torch.cumsum(p_target, dim=1)
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        if self.dist_r == 2:
            samplewise_emd = torch.sqrt(
                torch.mean(torch.pow(torch.abs(cdf_diff), 2), dim=1)
            )
        elif self.dist_r == 1:
            samplewise_emd = torch.mean(torch.abs(cdf_diff), dim=-1)
        return samplewise_emd.mean()
