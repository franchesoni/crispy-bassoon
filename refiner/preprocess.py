import json
from pathlib import Path
import tqdm
from pycocotools import mask as mask_utils
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from torchvision.transforms import v2 as transforms
from PIL import Image
import numpy as np

class SamDatasetOnline(torch.utils.data.Dataset):
    def __init__(self, samdata_dir, out_size=(256,256), split='train'):
        samdata_dir = Path(samdata_dir)
        self.split = split
        assert self.split in ['train', 'val', 'trainval']
        self.samdata_dir = samdata_dir
        print('split:', self.split, 'out_size:', out_size)
        self.out_size = out_size if self.split == 'train' else (644, 644)
        self.image_names = sorted([f for f in samdata_dir.glob('sa_*.jpg')])
        self.label_names = sorted([f for f in samdata_dir.glob('sa_*.json')])
        if self.split == 'train':
            self.image_names = self.image_names[:-min(len(self.image_names)//10, 1000)]
            self.label_names = self.label_names[:-min(len(self.label_names)//10, 1000)]
        elif self.split == 'val':
            self.image_names = self.image_names[-min(len(self.image_names)//10, 1000):]
            self.label_names = self.label_names[-min(len(self.label_names)//10, 1000):]
        elif self.split == 'trainval':
            pass  # use all
        assert all([img.stem == label.stem for img, label in zip(self.image_names, self.label_names)])
        self.num_samples = len(self.image_names)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        img = Image.open(self.image_names[idx])
        img = self.transform(img)
        anns = json.load(open(self.label_names[idx]))['annotations']

        # randomly choose only 10% of the masks
        n_masks = max(len(anns) // 10, min(5, len(anns)))
        assert n_masks > 0
        np.random.seed(idx)
        anns = np.random.choice(anns, n_masks)
        masks = [mask_utils.decode(ann['segmentation']) for ann in anns]
        mask = np.sum(masks, axis=0) > 0
        mask = torch.from_numpy(mask)  # bool

        if self.split == 'train':
            # random crop
            i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.1, 1.0), ratio=(0.7, 1.3))
            img = transforms.functional.crop(img, i, j, h, w)
            mask = transforms.functional.crop(mask, i, j, h, w)

        # now generate the classified patches as the downsampling of mask to (46,46)
        patches = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(46,46), mode='bilinear')
        patches = (patches > 0.5)
        # and upsample the patches to the output size
        patches = torch.nn.functional.interpolate(patches.float(), size=self.out_size, mode='bilinear')[0]

        # resize everythign to the output size
        imgmask = torch.nn.functional.interpolate(torch.cat((img, mask[None]), axis=0)[None], size=self.out_size, mode='bilinear').squeeze()
        img, mask = imgmask[:-1], imgmask[-1]
        return self.image_names[idx], img, patches, mask

def preprocess(sam_dataset_dir):
    # Define the function to process each sample
    def process_and_save(sample, sam_dataset_dir):
        imgname, img, patches, mask = sample
        Image.fromarray(255 * patches[0].numpy()).convert('L').save(sam_dataset_dir / (imgname.stem + '_coarse.png'))
        Image.fromarray(255 * mask.numpy()).convert('L').save(sam_dataset_dir / (imgname.stem + '_gt.png'))

    # Directory and dataset setup
    ds = SamDatasetOnline(sam_dataset_dir, split='trainval')

    # Set up the ThreadPoolExecutor
    num_workers = 10  # Or however many threads you want to use; often set to the number of cores
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        missing_indices = []
        for idx in range(len(ds)):
            imgname = ds.image_names[idx]
            dst_names = [sam_dataset_dir / (imgname.stem + '_gt.png'), sam_dataset_dir / (imgname.stem + '_coarse.png')]
            if not all([dst_name.exists() for dst_name in dst_names]):
                missing_indices.append(idx)

        # Create a future to process each sample in the dataset
        futures = [executor.submit(process_and_save, ds[idx], sam_dataset_dir) for idx in missing_indices]

        # Use tqdm to create a progress bar for the futures as they complete
        for future in tqdm.tqdm(as_completed(futures), total=len(ds)):
            future.result()  # This will raise any exceptions that occurred during execution

