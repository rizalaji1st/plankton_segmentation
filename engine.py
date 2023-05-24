import torch
from tqdm import tqdm

from metric import dice_coefficient, iou_score
from util import plot_train_progress


def train_fn(train_loader, model, optimizer, loss_fn, DEVICE):
    # Training
    model.train()
    train_loss = 0.0
    train_dice = 0.0
    train_iou = 0.0
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
            train_loss += loss.item() * images.size(0)
            train_dice += dice.item() * images.size(0)
            train_iou += iou.item() * images.size(0)

        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item(), dice=dice.item())

    # Calculate average training loss and dice score
    train_loss = train_loss / len(train_loader.dataset)
    train_dice = train_dice / len(train_loader.dataset)
    train_iou = train_iou / len(train_loader.dataset)
    return train_loss, train_dice, train_iou


def evaluate_fn(model, test_loader, loss_fn, DEVICE, PLOT_IMAGE_DURING_TRAINING=False):
    # Evaluation
    model.eval()
    test_loss = 0.0
    test_dice = 0.0
    test_iou = 0.0
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
                test_loss += loss.item() * images.size(0)
                test_dice += dice.item() * images.size(0)
                test_iou += iou.item() * images.size(0)

            loop.set_postfix(loss=loss.item(), dice=dice.item())

        # show prediction on test
        if PLOT_IMAGE_DURING_TRAINING:
            plot_train_progress(model, test_loader, DEVICE)

    # Calculate average test loss and dice score
    test_loss = test_loss / len(test_loader.dataset)
    test_dice = test_dice / len(test_loader.dataset)
    test_iou = test_iou / len(test_loader.dataset)
    return test_loss, test_dice, test_iou
