# train refiner from predicted patches to ground truth masks
import tqdm
from pathlib import Path
import random
from PIL import Image

import numpy as np

from torchvision.transforms import v2 as transforms
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SamDataset(torch.utils.data.Dataset):
    def __init__(self, samdata_dir, split='train'):
        samdata_dir = Path(samdata_dir)
        self.split = split
        assert self.split in ['train', 'val', 'trainval']
        self.samdata_dir = samdata_dir
        self.image_names = sorted([f for f in samdata_dir.glob('sa_*.jpg')])
        self.gt_names = sorted([f for f in samdata_dir.glob('sa_*gt.png')])
        self.patch_names = sorted([f for f in samdata_dir.glob('sa_*coarse.png')])
        if self.split == 'train':
            self.image_names = self.image_names[:-min(len(self.image_names)//10, 1000)]
            self.gt_names = self.gt_names[:-min(len(self.gt_names)//10, 1000)]
            self.patch_names = self.patch_names[:-min(len(self.patch_names)//10, 1000)]
        elif self.split == 'val':
            self.image_names = self.image_names[-min(len(self.image_names)//10, 1000):]
            self.gt_names = self.gt_names[-min(len(self.gt_names)//10, 1000):]
            self.patch_names = self.patch_names[-min(len(self.patch_names)//10, 1000):]
        elif self.split == 'trainval':
            pass

        assert all([img.stem.split('_')[:2] == gt.stem.split('_')[:2] == patch.stem.split('_')[:2] for img, gt, patch in zip(self.image_names, self.gt_names, self.patch_names)])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
 
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img = Image.open(self.image_names[idx])
        img = self.transform(img)
        patches = Image.open(self.patch_names[idx])
        mask = Image.open(self.gt_names[idx])
        return img, patches, mask


 

def minmaxnorm(x):
    return (x - x.min()) / (x.max() - x.min())


class Decoder(torch.nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.n_layers = n_layers
        self.initial_conv = torch.nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.conv = torch.nn.ModuleList([torch.nn.Conv2d(64, 64, kernel_size=3, padding=1) for _ in range(n_layers-1)])
        self.act = torch.nn.GELU()
        # LayerNorm should be properly defined for each conv layer or use another normalization
        self.norms = torch.nn.ModuleList([torch.nn.BatchNorm2d(64) for _ in range(n_layers-1)])
        self.final_conv = torch.nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.act(self.initial_conv(x))
        for i in range(self.n_layers-1):
            residual = x
            x = self.conv[i](x)
            x = self.act(x)
            x = self.norms[i](x)
            x += residual
        x = self.final_conv(x)
        return x

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# EarlyStopping class for convenience
class EarlyStopping:
    def __init__(self, patience, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def validate(model, val_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
        for images, patches, gt_masks in val_loader:
            images, patches, gt_masks = images.to(device), patches.to(device), gt_masks.to(device)
            model_input = torch.cat((images, patches), dim=1)
            predictions = model(model_input)
            loss = criterion(predictions, gt_masks.unsqueeze(1))
            val_loss += loss.item()
    return val_loss / len(val_loader)

def train(samdata_dir, seed=0, batch_size=16):
    seed_everything(seed)
    # Assuming 'Decoder' is your model and 'SamDataset' is your dataset
    # Define your dataset
    samdata_dir = Path(samdata_dir)
    train_ds = SamDataset(samdata_dir, split='train')
    val_ds = SamDataset(samdata_dir, split='val')


    # Create a DataLoader
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Instantiate the model
    model = Decoder(n_layers=5)

    # Send the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define an optimizer
    optimizer = Adam(model.parameters(), lr=1e-4)

    # Define a loss function
    criterion = BCEWithLogitsLoss()

    # Define the learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=2, min_delta=0.001)
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        train_dl_wrapped = tqdm.tqdm(train_dl, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for batch_idx, (images, patches, gt_masks) in enumerate(train_dl_wrapped):
            # Send data to device
            images, patches, gt_masks = images.to(device), patches.to(device), gt_masks.to(device)

            # Concatenate images and patches to form input to the model
            model_input = torch.cat((images, patches), dim=1)

            # Forward pass
            predictions = model(model_input)

            # Compute loss
            loss = criterion(predictions, gt_masks.unsqueeze(1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update weights
            optimizer.step()

            # Update train loss
            train_loss += loss.item()
            train_dl_wrapped.set_postfix({'train_loss': train_loss/(batch_idx+1)}, refresh=True)


        val_loss = validate(model, val_dl, criterion, device)

        # Step the scheduler on validation loss
        lr_scheduler.step(val_loss)

        # Call early stopping logic
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        # Save the model if this is the best epoch
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        # Check the learning rate and break if it's below a certain threshold
        for param_group in optimizer.param_groups:
            if param_group['lr'] < 1e-7:
                print(f"Learning rate very low, stopping training. Current LR: {param_group['lr']}")
                break

        # Display the validation loss on the progress bar
        train_dl_wrapped.set_postfix({"train_loss": train_loss / len(train_dl), "val_loss": val_loss})

    # After training, load the best model for inference or further training
    model.load_state_dict(torch.load('best_model.pth'))



if __name__ == '__main__':
    # get dataset dir from args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True)
    args = parser.parse_args()
    sam_dataset_dir = Path(args.datadir)
    train(sam_dataset_dir)

