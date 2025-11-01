import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from celeba_dataset import CelebADataset # Import our custom class from dataset.py

def get_dataloaders(root_dir, batch_size, image_size, random_seed):
    """
    Creates and returns train, validation, and test dataloaders
    by splitting the full dataset.
    """

    # 1. Define Image Transformations
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(), # Converts to [0, 1] range
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizes to [-1, 1] range
    ])

    # 2. Create Full Dataset
    print(f"\nLoading dataset from: {root_dir}")
    try:
        full_dataset = CelebADataset(root_dir=root_dir, transform=transform)
        total_size = len(full_dataset)

        if total_size == 0:
            print(f"[Error] Dataset is empty.")
            return None, None, None
        else:
            print(f"Successfully loaded {total_size} total images.")

        # 3. Calculate split sizes (80% train, 10% val, 10% test)
        train_size = int(total_size * 0.8)
        val_size = int(total_size * 0.1)
        # Ensure all splits add up to the total size
        test_size = total_size - train_size - val_size

        print(f"Splitting dataset into:")
        print(f"  Train: {train_size} images")
        print(f"  Validation: {val_size} images")
        print(f"  Test: {test_size} images")

        # 4. Split the dataset using a fixed seed for reproducibility
        generator = torch.Generator().manual_seed(random_seed)
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=generator
        )

        # 5. Create DataLoaders
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

    except Exception as e:
        print(f"\n[An error occurred during dataset loading]: {e}")
        return None, None, None