#src/dataset.py - FIXED VERSION

import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from src.utils import rgb_to_lab, get_historical_color_palette
import random

class HistoricalColorizationDataset(Dataset):
    """
    Dataset class for historical image colorization
    Loads color images, converts to LAB, and creates B&W input with color target
    """
    
    def __init__(self, image_dir, period='default', transform=None, image_size=256):
        """
        Args:
            image_dir: Directory containing color historical images
            period: Historical period for color palette reference
            transform: Optional transform to be applied to images
            image_size: Target size for images (default 256x256)
        """
        self.image_dir = image_dir
        self.period = period
        self.image_size = image_size
        self.color_palette = get_historical_color_palette(period)
        
        # Get all image files
        self.image_files = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for file in os.listdir(image_dir):
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                self.image_files.append(file)
        
        print(f"Found {len(self.image_files)} images for period: {period}")
        
        # Define transforms
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        # Additional transforms for data augmentation
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.RandomRotation(degrees=5),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict containing:
            - 'L': L channel (grayscale) as input [1, H, W]
            - 'ab': ab channels as target [2, H, W] 
            - 'rgb': original RGB for reference [3, H, W]
            - 'period': historical period info
        """
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a random other image if current one fails
            return self.__getitem__(random.randint(0, len(self.image_files) - 1))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Apply data augmentation randomly during training
        if random.random() > 0.5:
            image = self.augment_transform(image)
        
        # Convert RGB to LAB color space
        lab_image = rgb_to_lab(image)  # [3, H, W] -> LAB format
        
        # Split LAB channels
        L_channel = lab_image[0:1, :, :]  # Lightness [1, H, W]
        ab_channels = lab_image[1:3, :, :] # Color channels [2, H, W]
        
        # Normalize LAB values for training
        L_channel = L_channel / 50.0 - 1.0  # Normalize L to [-1, 1]
        ab_channels = ab_channels / 128.0     # Normalize ab to [-1, 1]
        
        return {
            'L': L_channel.float(),
            'ab': ab_channels.float(), 
            'rgb': image.float(),
            'period': self.period,
            'filename': self.image_files[idx]
        }

class InferenceDataset(Dataset):
    """
    Dataset for inference - takes grayscale images and prepares them for colorization
    """
    
    def __init__(self, image_dir, period='default', image_size=256):
        self.image_dir = image_dir
        self.period = period
        self.image_size = image_size
        
        # Get all grayscale image files
        self.image_files = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for file in os.listdir(image_dir):
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                self.image_files.append(file)
        
        print(f"Found {len(self.image_files)} images for inference")
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        
        # Load and convert to grayscale if needed
        image = Image.open(img_path).convert('L')  # Force grayscale
        
        # Convert to 3-channel for consistency
        image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Convert to LAB and extract L channel
        lab_image = rgb_to_lab(image)
        L_channel = lab_image[0:1, :, :]
        
        # Normalize L channel
        L_channel = L_channel / 50.0 - 1.0
        
        return {
            'L': L_channel.float(),
            'period': self.period,
            'filename': self.image_files[idx],
            'original_rgb': image.float()
        }

def create_dataloaders(train_dir, val_dir, period='default', batch_size=16, num_workers=4, image_size=256):
    """
    Create training and validation dataloaders
    
    Args:
        train_dir: Path to training images
        val_dir: Path to validation images  
        period: Historical period
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        image_size: Target image size
    
    Returns:
        train_loader, val_loader: DataLoader objects
    """
    
    # Create datasets
    train_dataset = HistoricalColorizationDataset(
        image_dir=train_dir,
        period=period,
        image_size=image_size
    )
    
    val_dataset = HistoricalColorizationDataset(
        image_dir=val_dir,
        period=period,
        image_size=image_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader

def create_inference_dataloader(image_dir, period='default', batch_size=1, image_size=256):
    """Create dataloader for inference on grayscale images"""
    
    inference_dataset = InferenceDataset(
        image_dir=image_dir,
        period=period,
        image_size=image_size
    )
    
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return inference_loader