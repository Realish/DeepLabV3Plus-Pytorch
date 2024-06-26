import os
import numpy as np
import h5py
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

def coco_cmap(N=256, normalized=False):
    """
    Create an N x 3 colormap for visualizing segmentation masks. Each unique value corresponds to a different color.
    """
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= bitget(c, 0) << (7 - j)
            g |= bitget(c, 1) << (7 - j)
            b |= bitget(c, 2) << (7 - j)
            c >>= 3
        cmap[i] = np.array([r, g, b])
    cmap = cmap / 255 if normalized else cmap
    return cmap

class COCOSegmentation(Dataset):
    def __init__(self, root, year='2017', image_set='train', transform=None):
        self.root = root
        self.year = year
        self.image_set = image_set
        self.transform = transform

        self.coco = COCO(os.path.join(root, 'annotations',
                                      f'instances_{image_set}{year}.json'))
        self.image_ids = self.coco.getImgIds()
        self.cmap = coco_cmap()
        
        self.images_dir = os.path.join(root, f'{image_set}{year}')

        if not os.path.isdir(self.images_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders are present.')
        

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.image_ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        image_info = coco.loadImgs(img_id)[0]

        # Load the image
        path = os.path.join(self.images_dir, image_info['file_name'])
        image = Image.open(path).convert('RGB')

        # Generate a mask image
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        for ann in anns:
            if ann['iscrowd'] == 0:  # Skip crowd annotations
                mask[coco.annToMask(ann) > 0] = ann['category_id']

        mask = Image.fromarray(mask, mode='L')

        if self.transform:
            image, mask = self.transform(image, mask)

        # Optionally, convert masks to categorical (one-hot encoding)
        # mask = torch.eye(len(self.cmap))[mask.long()]

        return image, mask

    @staticmethod
    def decode_target(mask):
        """Decode semantic mask to RGB image"""
        cmap = coco_cmap()
        return cmap[mask]
    

# Adjust the transform function to work directly on both image and mask
# This transform function is optional and can be customized as needed
def my_transforms(image_size=(512, 512)):
    """Transforms for the input image and mask."""
    return Compose([
        Resize(image_size),  # Resize both image and mask
        ToTensor(),  # Convert both image and mask to PyTorch tensors
        # Normalize only the image, if necessary
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
