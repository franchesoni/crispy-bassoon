import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import random
from PIL import Image
import cProfile

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



        # Get all possible unique identifiers from the image names
        unique_ids = sorted({p.stem.split('_img')[0] for p in samdata_dir.glob('sa_*_img.png')})

        # Initialize empty lists for images, ground truths, and coarse images
        matched_image_names = []
        matched_gt_names = []
        matched_coarse_names = []

        # Check for corresponding files and populate the lists
        for unique_id in unique_ids:
            img_name = samdata_dir / f'{unique_id}_img.png'
            gt_name = samdata_dir / f'{unique_id}_gt.png'
            coarse_name = samdata_dir / f'{unique_id}_coarse.png'

            if img_name.exists() and gt_name.exists() and coarse_name.exists():
                matched_image_names.append(img_name)
                matched_gt_names.append(gt_name)
                matched_coarse_names.append(coarse_name)

        self.image_names = sorted(matched_image_names)
        self.gt_names = sorted(matched_gt_names)
        self.coarse_names = sorted(matched_coarse_names)
        
        if self.split == 'train':
            self.image_names = self.image_names[:-min(len(self.image_names)//10, 1000)]
            self.gt_names = self.gt_names[:-min(len(self.gt_names)//10, 1000)]
            self.coarse_names = self.coarse_names[:-min(len(self.coarse_names)//10, 1000)]
        elif self.split == 'val':
            self.image_names = self.image_names[-min(len(self.image_names)//10, 1000):]
            self.gt_names = self.gt_names[-min(len(self.gt_names)//10, 1000):]
            self.coarse_names = self.coarse_names[-min(len(self.coarse_names)//10, 1000):]
        elif self.split == 'trainval':
            pass

        assert all([img.stem.split('_')[:2] == gt.stem.split('_')[:2] == coarse.stem.split('_')[:2] for img, gt, coarse in zip(self.image_names, self.gt_names, self.coarse_names)])

        self.totensor = transforms.ToTensor()
 
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img = Image.open(self.image_names[idx])
        img = self.totensor(img) / 128 - 0.5
        coarse = self.totensor(Image.open(self.coarse_names[idx]))
        mask = self.totensor(Image.open(self.gt_names[idx]))
        return img, coarse, mask


 

def minmaxnorm(x):
    return (x - x.min()) / (x.max() - x.min())


class Decoder(torch.nn.Module):
    def __init__(self, n_layers=3, hidden_size=64, in_channels=4):
        super().__init__()
        self.n_layers = n_layers
        self.initial_conv = torch.nn.Conv2d(in_channels, hidden_size, kernel_size=3, padding=1)
        self.conv = torch.nn.ModuleList([torch.nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1) for _ in range(n_layers-1)])
        self.act = torch.nn.GELU()
        # LayerNorm should be properly defined for each conv layer or use another normalization
        self.norms = torch.nn.ModuleList([torch.nn.BatchNorm2d(hidden_size) for _ in range(n_layers-1)])
        self.final_conv = torch.nn.Conv2d(hidden_size, 1, kernel_size=1)

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


def visualize(samdata_dir, seed=0):
    seed_everything(seed)
    # Assuming 'Decoder' is your model and 'SamDataset' is your dataset
    # Define your dataset
    samdata_dir = Path(samdata_dir)
    val_ds = SamDataset(samdata_dir, split='val')

    # Instantiate the model
    model = Decoder(n_layers=5, hidden_size=64)

    # Send the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if Path('best_model.pth').exists():
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    # Define a loss function
    criterion = BCEWithLogitsLoss()

    dstdir = Path('tmp/visu')
    dstdir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for idx in range(10):
            sample = val_ds[idx]
            images, coarse, gt_masks = sample[0][None], sample[1][None], sample[2][None]
            images, coarse, gt_masks = images.to(device), coarse.to(device), gt_masks.to(device)
            # Concatenate images and coarse to form input to the model
            model_input = torch.cat((images, coarse), dim=1)
            # Forward pass
            predictions = model(model_input)
            # Compute loss
            loss = criterion(predictions, gt_masks).item()

            dstfile = dstdir / f'bigimg_{idx}_loss_{loss}.png'
            save_sample(dstfile, images[0], coarse[0], gt_masks[0], predictions[0], loss)

def save_sample(dstfile, img, coarse, gt, pred, loss): 
    img, coarse, gt, pred = img.cpu(), coarse.cpu(), gt.cpu(), pred.cpu()
    bigimg = torch.cat((torch.cat((minmaxnorm(img), minmaxnorm(gt).repeat(3,1,1)), axis=2), torch.cat((minmaxnorm(coarse).repeat(3,1,1), minmaxnorm(pred).repeat(3,1,1)), axis=2)), axis=1).permute(1,2,0).numpy()
    plt.imsave(dstfile, bigimg)

if __name__ == '__main__':
    # get dataset dir from args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True)
    args = parser.parse_args()
    sam_dataset_dir = Path(args.datadir)
    visualize(sam_dataset_dir)

