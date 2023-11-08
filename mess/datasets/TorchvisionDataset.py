
from detectron2.data import DatasetCatalog, MetadataCatalog
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose, ToTensor


class TorchvisionDataset(Dataset):
    def __init__(self, dataset_name, transform, mask_transform=None, max_size=None, *args, **kwargs):
        self.dataset_name = dataset_name
        self.dataset = DatasetCatalog.get(dataset_name)
        self.max_size = max_size
        self.transform = transform
        self.mask_transform = mask_transform or Compose([ToTensor()])

        self.class_names = get_class_names(dataset_name)
        self.num_classes = len(self.class_names)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        # read image and mask
        input = self.dataset[item]
        image = Image.open(input['file_name']).convert('RGB')
        mask = Image.open(input['sem_seg_file_name'])
        # if not the same size, crop the image and mask to the min of their sizes
        if image.size != mask.size:
            RuntimeWarning(f'Image and mask {input["file_name"]} have different sizes. Cropping to the min of their sizes.')
            minw, minh = min(image.size[0], mask.size[0]), min(image.size[1], mask.size[1])
            min_size = (minw, minh)
            image = image.crop((0, 0, min_size[0], min_size[1]))
            mask = mask.crop((0, 0, min_size[0], min_size[1]))
    

        # resize image and mask if necessary while keeping aspect ratio
        if self.max_size is not None and max(image.size) > self.max_size:
            ratio = self.max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.BILINEAR)
            mask = mask.resize(new_size, Image.NEAREST)

        # apply transform
        image = self.transform(image)
        mask = self.mask_transform(mask)

        return image, mask


def get_class_names(name):
    return MetadataCatalog.get(name).stuff_classes


def get_detectron2_datasets():
    return list(DatasetCatalog.data)
