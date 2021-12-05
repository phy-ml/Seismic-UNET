import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from diffencoder.model import UNET
from diffencoder.utility.utils import *


# Hyperparameter
learning_rate = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 11
#set_deterministic()
set_all_seeds(seed=seed)
batch_size = 1
num_epochs = 10
num_workers = 2
pin_memory = True
load_model = False
train_seis = r'C:\Users\Syahrir Ridha\PycharmProjects\DiffEncoder\diffencoder\data\train-20210921T145844Z-001\train\seis'
train_mask = r'C:\Users\Syahrir Ridha\PycharmProjects\DiffEncoder\diffencoder\data\train-20210921T145844Z-001\train\fault\fault'
val_seis = r'C:\Users\Syahrir Ridha\PycharmProjects\DiffEncoder\diffencoder\data\validation-20210921T145852Z-001\validation\seis'
val_mask = r'C:\Users\Syahrir Ridha\PycharmProjects\DiffEncoder\diffencoder\data\validation-20210921T145852Z-001\validation\fault'

def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, target) in enumerate(loop):
        data = torch.autograd.Variable(data).to(DEVICE)
        # target = torch.autograd.Variable(target).float().unsqueeze(1).to(DEVICE)
        target = torch.autograd.Variable(target).float().to(DEVICE)

        # forward loop
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, target)

        # backpropagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm
        loop.set_postfix(loss=loss.item())

def main():

    # initialize the model
    model = UNet().to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optiimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, val_loader = get_loader(
        train_dir=train_seis,
        train_mask_dir=train_mask,
        val_dir=val_seis,
        val_mask_dir=val_mask,
        batch_size=batch_size,
        train_transform=None,
        val_transform=None,
        #num_workers=None,
        pin_memory=pin_memory
    )

    if load_model:
        load_checkpoint(torch.load('my_checkpoint.pth.tar'), model)

    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        train(train_loader, model, optiimizer, loss_fn, scaler)

        # save the model
        checkpoint = {
            "state_dict":model.state_dict(),
            "optimizer":optiimizer.state_dict()
        }
        save_checkpoint(checkpoint)

        check_accuracy(val_loader, model, DEVICE)

        save_preditions_as_image(val_loader, model, folder='preds/',device=DEVICE)

if __name__ == '__main__':
    main()
