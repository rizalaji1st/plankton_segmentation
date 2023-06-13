import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import torchvision.utils as vutils


def plot_pred_img(image, mask, pred):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 6))
    fig.tight_layout()

    ax1.axis('off')
    ax1.set_title('input image')
    ax1.imshow(np.transpose(vutils.make_grid(image, padding=2).cpu().numpy(),
                            (1, 2, 0)))

    ax2.axis('off')
    ax2.set_title('input mask')
    ax2.imshow(np.transpose(vutils.make_grid(mask, padding=2).cpu().numpy(),
                            (1, 2, 0)), cmap='gray')

    ax3.axis('off')
    ax3.set_title('predicted mask')
    ax3.imshow(np.transpose(vutils.make_grid(pred, padding=2).cpu().numpy(),
                            (1, 2, 0)), cmap='gray')

    plt.show()


def plot_train_progress(model, data_loader, device):
    image, mask = next(iter(data_loader))
    model = model.to(device)
    image = image.to(device)
    mask = mask.to(device)

    pred = model(image)
    plot_pred_img(image, mask, pred.detach())


def plot_performance_form_dict(metrics):
    df_metrics = pd.DataFrame.from_dict(metrics)
    train_dice_values = df_metrics['train_dice'].values
    test_dice_values = df_metrics['test_dice'].values
    train_iou_values = df_metrics['train_iou'].values
    test_iou_values = df_metrics['test_iou'].values
    train_precision_values = df_metrics['train_precision'].values
    test_precision_values = df_metrics['test_precision'].values
    train_recall_values = df_metrics['train_recall'].values
    test_recall_values = df_metrics['test_recall'].values
    train_pixel_accuracy_values = df_metrics['train_pixel_accuracy'].values
    test_pixel_accuracy_values = df_metrics['test_pixel_accuracy'].values
    train_loss_values = df_metrics['train_loss'].values
    test_loss_values = df_metrics['test_loss'].values

    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(8, 16))

    ax1.plot(train_dice_values, label='Train dice')
    ax1.plot(test_dice_values, label='Test dice')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('dice')
    ax1.set_title('Train and Test dice')
    ax1.legend()

    ax2.plot(train_iou_values, label='Train iou')
    ax2.plot(test_iou_values, label='Test iou')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('iou')
    ax2.set_title('Train and Test iou')
    ax2.legend()

    ax3.plot(train_loss_values, label='Train Loss')
    ax3.plot(test_loss_values, label='Test Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Train and Test Loss')
    ax3.legend()

    ax4.plot(train_precision_values, label='Train Precision')
    ax4.plot(test_precision_values, label='Test Precision')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Precision')
    ax4.set_title('Train and Test Precision')
    ax4.legend()
    
    ax5.plot(train_recall_values, label='Train Recall')
    ax5.plot(test_recall_values, label='Test Recall')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Recall')
    ax5.set_title('Train and Test Recall')
    ax5.legend()
    
    ax6.plot(train_pixel_accuracy_values, label='Train Pixel Accuracy')
    ax6.plot(test_pixel_accuracy_values, label='Test Pixel Accuracy')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Pixel Accuracy')
    ax6.set_title('Train and Test Pixel Accuracy')
    ax6.legend()

    plt.tight_layout()
    plt.show()


def save_experiment(SAVE_DIR, EXPERIMENT_NAME, model, metrics):
    EXPERIMENT_DIR = os.path.join(SAVE_DIR, EXPERIMENT_NAME)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    torch.save(model, os.path.join(EXPERIMENT_DIR, f"model_{EXPERIMENT_NAME}.pth"))
    torch.save(metrics, os.path.join(EXPERIMENT_DIR, f"metrics_{EXPERIMENT_NAME}.pth"))
    df_metrics = pd.DataFrame.from_dict(metrics)
    df_metrics.to_csv(os.path.join(EXPERIMENT_DIR, f"metrics_complete_{EXPERIMENT_NAME}.csv"), index=False)
    df_metrics.tail(1).to_csv(os.path.join(EXPERIMENT_DIR, f"metrics_last_{EXPERIMENT_NAME}.csv"), index=False)
