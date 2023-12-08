import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch
from loss import EMDLoss

def cal_metrics(output, target):
    bins = output.shape[-1]
    if bins == 10: # aes
        scores_mean = np.dot(output, np.arange(1, 11))
        labels_mean = np.dot(target, np.arange(1, 11))
    elif bins == 9: # aes
        scores_mean = np.dot(output, np.arange(1, 5.5, 0.5))
        labels_mean = np.dot(target, np.arange(1, 5.5, 0.5))
    elif bins == 5: # com, col, dof, lig, con
        scores_mean = np.dot(output, np.arange(1, 6))
        labels_mean = np.dot(target, np.arange(1, 6))
    elif bins == 1:
        scores_mean = output.squeeze()
        labels_mean = target.squeeze()

    srcc, _ = spearmanr(scores_mean, labels_mean)
    plcc, _ = pearsonr(scores_mean, labels_mean)
    mse = ((scores_mean - labels_mean)**2).mean()

    if bins == 10:
        diff = (((scores_mean-5.0) * (labels_mean-5.0)) >= 0)
    elif bins in [5, 9]:
        diff = (((scores_mean-3.0) * (labels_mean-3.0)) >= 0)
    acc = np.sum(diff) / len(scores_mean) * 100

    if bins != 1:
        output_tensor = torch.from_numpy(output)
        target_tensor = torch.from_numpy(target)
        with torch.no_grad():
            emd1 = EMDLoss(dist_r=1)(target_tensor, output_tensor).numpy().item()
            emd2 = EMDLoss(dist_r=2)(target_tensor, output_tensor).numpy().item()
    else:
        acc, emd1, emd2 = 0, 0, 0
                
    return [mse, srcc, plcc, acc, emd1, emd2]