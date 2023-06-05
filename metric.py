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

def get_pixel_accuracy(y_pred, y_true):
    y_pred = (y_pred > 0.5).float()
    correct_pixels = torch.sum(y_pred == y_true)
    total_pixels = y_pred.numel()
    accuracy = correct_pixels / total_pixels
    return accuracy

def get_precision(y_pred, y_true):
    y_pred = (y_pred > 0.5).float()
    true_positive = torch.sum((y_pred == 1) & (y_true == 1))
    false_positive = torch.sum((y_pred == 1) & (y_true == 0))
    precision = true_positive / (true_positive + false_positive + 1e-7)
    return precision


def get_recall(y_pred, y_true):
    y_pred = (y_pred > 0.5).float()
    true_positive = torch.sum((y_pred == 1) & (y_true == 1))
    false_negative = torch.sum((y_pred == 0) & (y_true == 1))
    recall = true_positive / (true_positive + false_negative + 1e-7)
    return recall