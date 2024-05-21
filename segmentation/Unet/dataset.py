import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        # Filter to include only image files and mask files
        self.images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        self.masks = [f for f in os.listdir(masks_dir) if f.endswith('.png')]

        # Sort images and masks to ensure they are aligned
        self.images.sort()
        self.masks.sort()

        # Debug prints to verify file paths and contents
        print(f"Found {len(self.images)} images and {len(self.masks)} masks")
        print(f"Image files: {self.images[:5]}")
        print(f"Mask files: {self.masks[:5]}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace('.jpg', '.png')
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        # Debug prints to verify file paths
        # print(f"Image path: {img_path}")
        # print(f"Mask path: {mask_path}")

        if not os.path.exists(img_path):
            print(f"Image file not found: {img_path}")
        if not os.path.exists(mask_path):
            print(f"Mask file not found: {mask_path}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = np.array(mask)
        mask[mask > 0] = 1
        mask = torch.tensor(mask, dtype=torch.float32)  # Ensure the mask is float32

        return image, mask



