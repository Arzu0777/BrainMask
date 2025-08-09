import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

def dice_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    smooth = 1e-5
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

def iou_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    smooth = 1e-5
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def recall_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    tp = (preds * targets).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))
    recall = tp / (tp + fn + 1e-5)
    return recall.mean()

def hausdorff_distance95(preds, targets):
    """
    Approximated 95th percentile Hausdorff distance.
    Assumes preds and targets are 4D tensors: (B, 1, H, W)
    """
    distances = []
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()

    for pred, target in zip(preds_np, targets_np):
        pred = pred.squeeze()
        target = target.squeeze()
        if pred.sum() == 0 or target.sum() == 0:
            continue
        dt_pred = distance_transform_edt(1 - pred)
        dt_target = distance_transform_edt(1 - target)
        sds = np.concatenate([dt_target[pred == 1], dt_pred[target == 1]])
        if len(sds) > 0:
            distances.append(np.percentile(sds, 95))
    return torch.tensor(distances).mean() if distances else torch.tensor(0.0)
