import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian(window_size, sigma):
    gauss = torch.Tensor([torch.exp(-torch.Tensor([(x - window_size//2)**2])/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def create_window(window_size, channel, device):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1).to(device)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True, device='cpu'):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.device = device
        self.window = create_window(window_size, self.channel, self.device)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel != self.channel:
            self.window = create_window(self.window_size, channel, img1.device).type(img1.dtype)
            self.channel = channel

        return _ssim(img1, img2, self.window, self.window_size, channel, self.size_average)

# And when you instantiate the SSIM in the ReconstructionLoss, pass the device:
class ReconstructionLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True, device='cpu'):
        super(ReconstructionLoss, self).__init__()
        self.ssim = SSIM(window_size, size_average, device)

    def forward(self, img1, img2):
        return 1 - self.ssim(img1, img2)


# In your main training script
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion_pretext = ReconstructionLoss(device=device).to(device)


    
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
        predicted_scores = predicted_scores.float()*10 #0130 update, sigmoid so times 10
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
