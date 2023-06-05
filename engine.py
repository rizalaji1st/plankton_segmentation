import torch
from tqdm import tqdm
import timeit

from plankton_segmentation.metric import dice_coefficient, iou_score, get_pixel_accuracy, get_precision, get_recall
from plankton_segmentation.util import plot_train_progress
# from metric import dice_coefficient, iou_score
# from util import plot_train_progress


def train_fn(train_loader, model, optimizer, loss_fn, DEVICE):
    time_start = timeit.default_timer()
    # Training
    model.train()
    train_loss = 0.0
    train_dice = 0.0
    train_iou = 0.0
    train_pixel_accuracy = 0.0
    train_precision = 0.0
    train_recall = 0.0
    loop = tqdm(train_loader)
    for batch_idx, (images, masks) in enumerate(loop):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda'):
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            dice = dice_coefficient(outputs, masks)
            iou = iou_score(outputs, masks)
            pixel_accuracy = get_pixel_accuracy(outputs, masks)
            precision = get_precision(outputs, masks)
            recall = get_recall(outputs, masks)
            train_loss += loss.item() * images.size(0)
            train_dice += dice.item() * images.size(0)
            train_iou += iou.item() * images.size(0)
            train_pixel_accuracy += pixel_accuracy.item() * images.size(0)
            train_precision += precision.item() * images.size(0)
            train_recall += recall.item() * images.size(0)

        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item(), dice=dice.item())

    # Calculate average training loss and dice score
    train_loss = train_loss / len(train_loader.dataset)
    train_dice = train_dice / len(train_loader.dataset)
    train_iou = train_iou / len(train_loader.dataset)
    train_pixel_accuracy = train_pixel_accuracy / len(train_loader.dataset)
    train_precision = train_precision / len(train_loader.dataset)
    train_recall = train_recall / len(train_loader.dataset)
    time_stop = timeit.default_timer()
    train_time = time_stop - time_start
    return train_loss, train_dice, train_iou, train_pixel_accuracy, train_precision, train_recall, train_time


def evaluate_fn(model, test_loader, loss_fn, DEVICE, PLOT_IMAGE_DURING_TRAINING=False):
    time_start = timeit.default_timer()
    # Evaluation
    model.eval()
    test_loss = 0.0
    test_dice = 0.0
    test_iou = 0.0
    test_pixel_accuracy = 0.0
    test_precision = 0.0
    test_recall = 0.0
    with torch.no_grad():
        loop = tqdm(test_loader)
        for images, masks in loop:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            with torch.autocast(device_type='cuda'):
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                dice = dice_coefficient(outputs, masks)
                iou = iou_score(outputs, masks)
                pixel_accuracy = get_pixel_accuracy(outputs, masks)
                precision = get_precision(outputs, masks)
                recall = get_recall(outputs, masks)
                test_loss += loss.item() * images.size(0)
                test_dice += dice.item() * images.size(0)
                test_iou += iou.item() * images.size(0)
                test_pixel_accuracy += pixel_accuracy.item() * images.size(0)
                test_precision += precision.item() * images.size(0)
                test_recall += recall.item() * images.size(0)

            loop.set_postfix(loss=loss.item(), dice=dice.item())

        # show prediction on test
        if PLOT_IMAGE_DURING_TRAINING:
            plot_train_progress(model, test_loader, DEVICE)

    # Calculate average test loss and dice score
    test_loss = test_loss / len(test_loader.dataset)
    test_dice = test_dice / len(test_loader.dataset)
    test_iou = test_iou / len(test_loader.dataset)
    test_pixel_accuracy = test_pixel_accuracy / len(test_loader.dataset)
    test_precision = test_precision / len(test_loader.dataset)
    test_recall = test_recall / len(test_loader.dataset)
    time_stop = timeit.default_timer()
    test_time = time_stop - time_start
    return test_loss, test_dice, test_iou, test_pixel_accuracy, test_precision, test_recall, test_time
