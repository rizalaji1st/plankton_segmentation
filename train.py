import os.path

import segmentation_models_pytorch as smp
from pathlib import Path

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import nn, optim

from dataset import get_plankton_dataset_loader
from engine import train_fn
from model import get_unet_model

DATA_DIR = Path("")
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 3e-4
EPOCHS = 1

train_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    A.ColorJitter(p=0.2),
    A.HorizontalFlip(p=0.5),
    ToTensorV2(),
])

test_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    ToTensorV2(),
])
train_loader = get_plankton_dataset_loader(data_dir=DATA_DIR, batch_size=BATCH_SIZE,
                                           is_train=True, transform=train_transform)
test_loader = get_plankton_dataset_loader(data_dir=TEST_DIR, batch_size=BATCH_SIZE,
                                          is_train=False, transform=test_transform)
valid_loader = get_plankton_dataset_loader(data_dir=VALID_DIR, batch_size=BATCH_SIZE,
                                           is_train=False, transform=test_transform)

model = get_unet_model()
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    print('#'*20, str(epoch))
    train_fn(train_loader, model, optimizer, loss_fn, DEVICE)
