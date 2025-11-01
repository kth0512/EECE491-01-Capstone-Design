import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image

class CelebADataset(Dataset):
    """
    Custom Dataset class for CelebA.
    It reads images directly from a root folder,
    as required by our file structure.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Find all .jpg files and sort the list for consistency
        print(f"Searching for '*.jpg' files in: {root_dir}")
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, '*.jpg')))

        if len(self.image_paths) == 0:
            print(f"[Warning] No '.jpg' files found in {root_dir}.")
        else:
            print(f"Successfully found {len(self.image_paths)} images.")

    def __len__(self):
        # Return the total number of images in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load one image based on its index
        img_path = self.image_paths[idx]

        # Open image and ensure it's in RGB format (avoids 1-channel errors)
        image = Image.open(img_path).convert('RGB')

        # Apply transformations (e.g., resize, ToTensor, normalize)
        if self.transform:
            image = self.transform(image)

        # We don't need labels for this task, so return a dummy '0'
        label = 0
        return image, label