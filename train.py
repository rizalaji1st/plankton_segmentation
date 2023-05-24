# !pip install -q segmentation-models-pytorch
# !git clone https://github.com/kangPrayit/plankton_segmentation

from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch import optim, nn

# from plankton_segmentation.dataset import *
# from plankton_segmentation.model import *
from dataset import *
from model import *
from engine import *
from util import save_experiment


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(42)

DATA_DIR = Path('/content/drive/MyDrive/DEEP LEARNING PROJECT/PRAYITNO/datasets/plankton_cocov2')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
SAVE_DIR = Path('/content/drive/MyDrive/DEEP LEARNING PROJECT/PRAYITNO/experiments/')

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BATCH_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 3e-4
EPOCHS = 50
SAVE_EXPERIMENT = False
EXPERIMENT_NAME = "unet_efficientnet_b3_2_notebook"
PLOT_IMAGE_DURING_TRAINING = False

# Data Processing
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
train_loader = get_plankton_dataset_loader(data_dir=TRAIN_DIR, batch_size=BATCH_SIZE,
                                           is_train=True, transform=train_transform)
test_loader = get_plankton_dataset_loader(data_dir=TEST_DIR, batch_size=BATCH_SIZE,
                                          is_train=False, transform=test_transform)
valid_loader = get_plankton_dataset_loader(data_dir=VALID_DIR, batch_size=BATCH_SIZE,
                                           is_train=False, transform=test_transform)

# Training loop
model = get_unet_model()
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
metrics = {"train_loss": [], "test_loss": [],
           "train_dice": [], "test_dice": [],
           "train_iou": [], "test_iou": []}
model = model.to(DEVICE)
for epoch in range(EPOCHS):
    print('#*60', f"Epoch {epoch + 1} \n")
    train_loss, train_dice, train_iou = train_fn(train_loader, model, optimizer, loss_fn, DEVICE)
    test_loss, test_dice, test_iou = evaluate_fn(model, test_loader, loss_fn, DEVICE, PLOT_IMAGE_DURING_TRAINING)

    print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
          f"Train Dice: {train_dice:.4f}, Test Dice: {test_dice:.4f}, "
          f"Train iou: {train_iou:.4f}, Test iou: {test_iou:.4f} \n")

    metrics["train_loss"].append(train_loss)
    metrics["train_dice"].append(train_dice)
    metrics["train_iou"].append(train_iou)
    metrics["test_loss"].append(test_loss)
    metrics["test_dice"].append(test_dice)
    metrics["test_iou"].append(test_iou)

if SAVE_EXPERIMENT:
    save_experiment(SAVE_DIR, EXPERIMENT_NAME, model, metrics)
