import random
import torch
import numpy as np
import os
import torchvision
from diffencoder.data.load_seismic import Seismic_Dataset
from diffencoder.model.UNET import UNet
from torch.utils.data import DataLoader

#Cementing_82


def set_deterministic():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_deterministic(True)


def set_all_seeds(seed):
    os.environ['PL_GLOBAL_SEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_epoch_loss_autoencoder(model, data_loader, loss_fn, device):
    model.eval()
    curr_loss, num_example = 0.0, 0
    with torch.no_grad():
        for features, _ in data_loader:
            features = features.to(device)
            pred = model(features)
            loss = loss_fn(pred, features, reduction='sum')
            num_example += features.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_example
        return curr_loss


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print("=> Saving Checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading Checkpoint")
    model.load_sate_dict(checkpoint['state_dict'])


def get_loader(
        train_dir,
        train_mask_dir,
        val_dir,
        val_mask_dir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True
):
    # training dataset
    train_ds = Seismic_Dataset(
        data=train_dir,
        label=train_mask_dir,
        transform=train_transform
    )

    # training loader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        #num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    # validation dataset
    val_ds = Seismic_Dataset(
        data=val_dir,
        label=val_mask_dir,
        transform=val_transform
    )

    # validation loader
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        #num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device='cuda'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds *y).sum()) / ( (preds + y).sum() + 1e-8 )

    print(f'{num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}')
    print(f'Dice Score: {dice_score/len(loader)}')
    model.train()

def save_preditions_as_image(
        loader, model, folder, device='cuda'
):
    model.eval()
    for idx, (x,y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f'{folder}/pred_{idx}.png')
        torchvision.utils.save_image(y.unsqueeze(1), f'{folder}{idx}.png')

    model.train()

