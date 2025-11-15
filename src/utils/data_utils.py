import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from ..datasets.celeba_dataset import CelebADataset 

def get_transform(image_size):
    # Define image transformations
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(), # Converts to [0, 1] range
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizes to [-1, 1] range
    ])
    return transform 
    
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
        transform = get_transform(image_size)

        print(f"\nLoading dataset from: {root_dir}")
        full_dataset = CelebADataset(root_dir=root_dir, transform=transform)
        

        if len(full_dataset) == 0:
            raise Exception("Dataset is empty. Check the 'root_dir' path.")
        print(f"Successfully loaded {len(full_dataset)} total images.")

        train_dataset, val_dataset, test_dataset = split_dataset(full_dataset, 
                                                                 random_seed, 
                                                                 train_ratio=train_ratio, 
                                                                 val_ratio=val_ratio)

        return create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size)
    
    except Exception as e:
        print(f"\n[An error occurred during dataset loading]: {e}")
        return None, None, None