# src/utils.py - FIXED VERSION

import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from skimage import color
import os
import yaml

def rgb_to_lab(rgb_image):
    """
    Convert RGB image to LAB color space using PyTorch tensors
    Args:
        rgb_image: torch.Tensor of shape (B, 3, H, W) or (3, H, W)
    Returns:
        lab_image: torch.Tensor in LAB format
    """
    # Convert tensor to numpy for skimage processing
    if isinstance(rgb_image, torch.Tensor):
        rgb_np = rgb_image.permute(1, 2, 0).cpu().numpy()
    else:
        rgb_np = rgb_image
    
    # Ensure values are in [0, 1] range
    if rgb_np.max() > 1.0:
        rgb_np = rgb_np / 255.0
    
    # Convert to LAB
    lab_np = color.rgb2lab(rgb_np)
    
    # Convert back to tensor
    lab_tensor = torch.from_numpy(lab_np).permute(2, 0, 1).float()
    
    return lab_tensor

def lab_to_rgb(lab_image):
    """
    Convert LAB image back to RGB
    Args:
        lab_image: torch.Tensor in LAB format
    Returns:
        rgb_image: torch.Tensor in RGB format
    """
    if isinstance(lab_image, torch.Tensor):
        lab_np = lab_image.permute(1, 2, 0).cpu().numpy()
    else:
        lab_np = lab_image
    
    # Convert LAB to RGB
    rgb_np = color.lab2rgb(lab_np)
    
    # Convert back to tensor and ensure [0, 1] range
    rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).float()
    rgb_tensor = torch.clamp(rgb_tensor, 0, 1)
    
    return rgb_tensor

def get_historical_color_palette(period):
    """
    Define color palettes for different historical periods
    Args:
        period: str ('1920s', 'wwii', 'victorian', 'medieval')
    Returns:
        palette: dict with dominant colors for the period
    """
    palettes = {
        '1920s': {
            'dominant_colors': [(0.8, 0.6, 0.2), (0.2, 0.2, 0.2), (0.9, 0.8, 0.7)],  # Gold, Black, Cream
            'accent_colors': [(0.6, 0.2, 0.3), (0.2, 0.4, 0.6), (0.4, 0.6, 0.3)],    # Deep Red, Blue, Green
            'skin_tones': [(0.9, 0.7, 0.6), (0.8, 0.6, 0.5), (0.7, 0.5, 0.4)],
            'description': 'Art Deco era with metallic golds and rich jewel tones'
        },
        'wwii': {
            'dominant_colors': [(0.4, 0.3, 0.2), (0.5, 0.4, 0.2), (0.6, 0.5, 0.4)],  # Browns, Khaki
            'accent_colors': [(0.2, 0.3, 0.2), (0.3, 0.2, 0.1), (0.4, 0.4, 0.4)],    # Olive, Dark Brown, Grey
            'skin_tones': [(0.8, 0.6, 0.5), (0.7, 0.5, 0.4), (0.6, 0.4, 0.3)],
            'description': 'Wartime muted earth tones and military colors'
        },
        'victorian': {
            'dominant_colors': [(0.3, 0.1, 0.2), (0.1, 0.2, 0.4), (0.2, 0.3, 0.1)],  # Burgundy, Navy, Forest
            'accent_colors': [(0.6, 0.4, 0.2), (0.5, 0.3, 0.4), (0.4, 0.5, 0.6)],    # Bronze, Mauve, Steel
            'skin_tones': [(0.9, 0.8, 0.7), (0.8, 0.7, 0.6), (0.7, 0.6, 0.5)],
            'description': 'Rich, deep saturated colors with ornate details'
        },
        'default': {
            'dominant_colors': [(0.5, 0.5, 0.5), (0.3, 0.3, 0.3), (0.7, 0.7, 0.7)],
            'accent_colors': [(0.6, 0.4, 0.4), (0.4, 0.6, 0.4), (0.4, 0.4, 0.6)],
            'skin_tones': [(0.8, 0.6, 0.5), (0.7, 0.5, 0.4), (0.6, 0.4, 0.3)],
            'description': 'General historical color palette'
        }
    }
    
    return palettes.get(period.lower(), palettes['default'])

def save_comparison_image(original, colorized, save_path, title="Colorization Result"):
    """
    Save side-by-side comparison of original and colorized images - FIXED VERSION
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original (grayscale)
    if len(original.shape) == 3:
        original_gray = torch.mean(original, dim=0) if isinstance(original, torch.Tensor) else np.mean(original, axis=0)
    else:
        original_gray = original
    
    axes[0].imshow(original_gray, cmap='gray')  # FIXED: was axes.set_title
    axes[0].set_title('Original B&W')          # FIXED: proper indexing
    axes[0].axis('off')                        # FIXED: proper indexing
    
    # Colorized
    if isinstance(colorized, torch.Tensor):
        colorized_np = colorized.permute(1, 2, 0).cpu().numpy()
    else:
        colorized_np = colorized
    
    axes[1].imshow(colorized_np)               # FIXED: was axes[4]
    axes[1].set_title('Colorized')             # FIXED: was axes[4]
    axes[1].axis('off')                        # FIXED: was axes[4]
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_directories(base_path):
    """Create necessary directories for the project"""
    dirs = [
        'data/raw', 'data/processed', 'data/train', 'data/validation',
        'outputs/colorized', 'outputs/samples', 'models', 'checkpoints', 'logs'
    ]
    
    for dir_path in dirs:
        full_path = os.path.join(base_path, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {full_path}")

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded: epoch {epoch}, loss {loss:.4f}")
    return model, optimizer, epoch, loss