import torch


def dice_coefficient(y_pred, y_true):
    smooth = 1e-7
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float()
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def iou_score(y_pred, y_true):
    smooth = 1e-7
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float()
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou
