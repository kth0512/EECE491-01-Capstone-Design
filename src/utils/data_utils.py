import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from ..datasets.celeba_dataset import CelebADataset 

import os
import time

class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def get_transform(image_size):
    # Define image transformations
    
    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=15, 
            translate=(0.1, 0.1), 
            interpolation=transforms.InterpolationMode.BILINEAR
            ),
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.8, 1.0), # 80% of face should remain at least.
            ratio=(0.95, 1.05)
        ),
        transforms.RandomHorizontalFlip(p=0.5),

        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        
        transforms.ToTensor(), # Converts to [0, 1] range
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizes to [-1, 1] range
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor(), # Converts to [0, 1] range
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizes to [-1, 1] range
    ])
    return train_transform, val_transform
    
def split_dataset(full_dataset, random_seed, train_ratio, val_ratio):
    # Split the dataset using a fixed seed for reproducibility
    if train_ratio + val_ratio > 1.0 or train_ratio < 0 or val_ratio < 0:
        raise ValueError(f"Invalid split ratios: train={train_ratio}, val={val_ratio}")
        
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size =  int(total_size*val_ratio)
    test_size = total_size - train_size - val_size
    
    print(f"Splitting dataset into:")
    print(f"  Train: {train_size} images")
    print(f"  Validation: {val_size} images")
    print(f"  Test: {test_size} images")
    
    generator = torch.Generator().manual_seed(random_seed)
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
        )
    return train_dataset, val_dataset, test_dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    # Use num_workers > 0 and pin_memory=True for faster loading
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )

    print("\nDataLoaders created successfully.")
    return train_loader, val_loader, test_loader

def get_dataloaders(root_dir, batch_size, image_size, random_seed, train_ratio=0.8, val_ratio=0.1):
    
    try:
        train_transform, val_transform = get_transform(image_size)

        print(f"\nLoading dataset from: {root_dir}")
        full_dataset = CelebADataset(root_dir=root_dir, transform=None)
        

        if len(full_dataset) == 0:
            raise Exception("Dataset is empty. Check the 'root_dir' path.")
        print(f"Successfully loaded {len(full_dataset)} total images.")

        train_subset, val_subset, test_subset = split_dataset(full_dataset, 
                                                                 random_seed, 
                                                                 train_ratio=train_ratio, 
                                                                 val_ratio=val_ratio)
        
        train_dataset = TransformedSubset(train_subset, train_transform)
        val_dataset = TransformedSubset(val_subset, val_transform)
        test_dataset = TransformedSubset(test_subset, val_transform)

        return create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size)
    
    except Exception as e:
        print(f"\n[An error occurred during dataset loading]: {e}")
        return None, None, None
        
def prepare_dataset(drive_archive_path, local_archive_path, extract_path):
    """
    Copy and decompress the compressed file on Google Drive to the local runtime.
    return: directory path where the data is ready (str)
    """
    print("\nStarting data setup...")
    start_setup_time = time.time()
    
    local_data_dir = os.path.join(extract_path, "content", "cropped_celeba")

    if not os.path.exists(local_data_dir):
        print(f"Copying {drive_archive_path} to local runtime...")
        
        if not os.path.exists(drive_archive_path):
            raise FileNotFoundError(f"[FATAL ERROR] Source file not found: {drive_archive_path}")

        # 1. Copy
        os.system(f"cp '{drive_archive_path}' '{local_archive_path}'")
        print("Copy complete.")

        # 2. Untar
        print(f"Untarring {local_archive_path} to {extract_path}...")
        os.makedirs(extract_path, exist_ok=True)
        os.system(f"tar -xf '{local_archive_path}' -C '{extract_path}'")
        print("Untar complete.")

        # 3. Cleanup
        if os.path.exists(local_archive_path):
            os.remove(local_archive_path)
    else:
        print(f"Data directory {local_data_dir} already exists. Skipping copy/untar.")

    print(f"Data setup finished in {time.time() - start_setup_time:.2f} seconds.")

    # Sanity Check
    if not os.path.exists(local_data_dir):
        raise FileNotFoundError(f"[FATAL ERROR] The expected data directory does not exist: {local_data_dir}")
    
    print(f"Successfully found data at: {local_data_dir}")
    return local_data_dir