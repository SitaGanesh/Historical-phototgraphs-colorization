# src/train.py - FIXED VERSION

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
from tqdm import tqdm
import numpy as np

from src.model import create_model
from src.dataset import create_dataloaders
from src.utils import save_comparison_image, save_checkpoint, lab_to_rgb

class ColorizationLoss(nn.Module):
    """
    Custom loss function for colorization
    Combines L1 loss with perceptual considerations
    """
    
    def __init__(self, alpha=1.0, beta=0.1):
        super().__init__()
        self.alpha = alpha  # Weight for L1 loss
        self.beta = beta    # Weight for smoothness loss
        self.l1_loss = nn.L1Loss()
        
    def forward(self, pred_ab, target_ab):
        """
        Args:
            pred_ab: Predicted ab channels [B, 2, H, W]
            target_ab: Target ab channels [B, 2, H, W]
        """
        # Main L1 loss for color accuracy
        l1_loss = self.l1_loss(pred_ab, target_ab)
        
        # Smoothness loss to encourage realistic color transitions
        diff_x = torch.abs(pred_ab[:, :, 1:, :] - pred_ab[:, :, :-1, :])
        diff_y = torch.abs(pred_ab[:, :, :, 1:] - pred_ab[:, :, :, :-1])
        smoothness_loss = torch.mean(diff_x) + torch.mean(diff_y)
        
        total_loss = self.alpha * l1_loss + self.beta * smoothness_loss
        
        return total_loss, l1_loss, smoothness_loss

class HistoricalColorizationTrainer:
    """
    Trainer class for historical image colorization model
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = create_model(
            device=self.device,
            use_period_embedding=config.get('use_period_embedding', True)
        )
        
        # Create loss function
        self.criterion = ColorizationLoss(
            alpha=config.get('loss_alpha', 1.0),
            beta=config.get('loss_beta', 0.1)
        )
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.get('lr_step_size', 30),
            gamma=config.get('lr_gamma', 0.5)
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Create directories
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['sample_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)
        
        # Tensorboard writer
        self.writer = SummaryWriter(config['log_dir'])
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_smooth_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            L_channel = batch['L'].to(self.device)          # [B, 1, H, W]
            ab_target = batch['ab'].to(self.device)         # [B, 2, H, W]
            period_labels = batch['period']                 # List of strings
            
            # Forward pass
            ab_pred = self.model(L_channel, period_labels)
            
            # Calculate loss
            total_loss, l1_loss, smooth_loss = self.criterion(ab_pred, ab_target)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            epoch_loss += total_loss.item()
            epoch_l1_loss += l1_loss.item()
            epoch_smooth_loss += smooth_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'L1': f'{l1_loss.item():.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log to tensorboard every 100 batches
            if batch_idx % 100 == 0:
                global_step = self.current_epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', total_loss.item(), global_step)
                self.writer.add_scalar('Train/L1_Loss', l1_loss.item(), global_step)
                self.writer.add_scalar('Train/Smooth_Loss', smooth_loss.item(), global_step)
        
        # Calculate average losses
        avg_loss = epoch_loss / len(train_loader)
        avg_l1 = epoch_l1_loss / len(train_loader)
        avg_smooth = epoch_smooth_loss / len(train_loader)
        
        return avg_loss, avg_l1, avg_smooth
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        
        epoch_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_smooth_loss = 0.0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Validation Epoch {self.current_epoch}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move data to device
                L_channel = batch['L'].to(self.device)
                ab_target = batch['ab'].to(self.device)
                period_labels = batch['period']
                
                # Forward pass
                ab_pred = self.model(L_channel, period_labels)
                
                # Calculate loss
                total_loss, l1_loss, smooth_loss = self.criterion(ab_pred, ab_target)
                
                # Accumulate losses
                epoch_loss += total_loss.item()
                epoch_l1_loss += l1_loss.item()
                epoch_smooth_loss += smooth_loss.item()
                
                progress_bar.set_postfix({'Val Loss': f'{total_loss.item():.4f}'})
                
                # Save sample colorizations from first batch
                if batch_idx == 0:
                    self.save_sample_results(batch, ab_pred)
        
        # Calculate average losses
        avg_loss = epoch_loss / len(val_loader)
        avg_l1 = epoch_l1_loss / len(val_loader)
        avg_smooth = epoch_smooth_loss / len(val_loader)
        
        return avg_loss, avg_l1, avg_smooth
    
    def save_sample_results(self, batch, ab_pred):
        """Save sample colorization results - FIXED VERSION"""
        # Take first image from batch
        L_channel = batch['L'][0]          # [1, H, W]
        ab_target = batch['ab'][0]         # [2, H, W] - FIXED: was missing [0]
        ab_prediction = ab_pred[0]         # [2, H, W] - FIXED: was missing [0]
        original_rgb = batch['rgb'][0]     # [3, H, W] - FIXED: was missing [0]
        
        # Denormalize
        L_denorm = (L_channel + 1.0) * 50.0
        ab_target_denorm = ab_target * 128.0
        ab_pred_denorm = ab_prediction * 128.0
        
        # Create LAB images
        lab_target = torch.cat([L_denorm, ab_target_denorm], dim=0)
        lab_pred = torch.cat([L_denorm, ab_pred_denorm], dim=0)
        
        # Convert to RGB
        rgb_target = lab_to_rgb(lab_target)
        rgb_pred = lab_to_rgb(lab_pred)
        
        # Save comparison image
        save_path = os.path.join(
            self.config['sample_dir'], 
            f'epoch_{self.current_epoch:03d}_sample.png'
        )
        
        save_comparison_image(
            original_rgb, rgb_pred, save_path,
            title=f"Epoch {self.current_epoch} - Period: {batch['period'][0]}"
        )
    
    def train(self, train_loader, val_loader, num_epochs):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_l1, train_smooth = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_l1, val_smooth = self.validate_epoch(val_loader)
            
            # Learning rate step
            self.scheduler.step()
            
            # Record losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Log to tensorboard
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            self.writer.add_scalar('Epoch/Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f} (L1: {train_l1:.4f}, Smooth: {train_smooth:.4f})")
            print(f"Val Loss: {val_loss:.4f} (L1: {val_l1:.4f}, Smooth: {val_smooth:.4f})")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint if best validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, checkpoint_path
                )
                print(f"✓ Best model saved with validation loss: {val_loss:.4f}")
            
            # Save regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(
                    self.config['checkpoint_dir'], 
                    f'checkpoint_epoch_{epoch+1}.pth'
                )
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, checkpoint_path
                )
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Close tensorboard writer
        self.writer.close()

def main():
    """Main training function"""
    
    # Training configuration
    config = {
        # Data paths (adjust according to your structure)
        'train_data_dir': 'D:/colorization_task2/historical-colorization/colorization/data/train',
        'val_data_dir': 'D:/colorization_task2/historical-colorization/colorization/data/validation',
        
        # Model parameters
        'use_period_embedding': True,
        'image_size': 256,
        
        # Training parameters
        'batch_size': 8,  # Adjust based on your GPU memory
        'num_epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'lr_step_size': 30,
        'lr_gamma': 0.5,
        
        # Loss parameters
        'loss_alpha': 1.0,
        'loss_beta': 0.1,
        
        # Historical period
        'period': '1920s',  # Change as needed
        
        # Output directories
        'checkpoint_dir': 'D:/colorization_task2/historical-colorization/colorization/checkpoints',
        'sample_dir': 'D:/colorization_task2/historical-colorization/colorization/outputs/samples',
        'log_dir': 'D:/colorization_task2/historical-colorization/colorization/logs',
        
        # Data loading
        'num_workers': 4
    }
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_dir=config['train_data_dir'],
        val_dir=config['val_data_dir'],
        period=config['period'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        image_size=config['image_size']
    )
    
    # Create trainer
    trainer = HistoricalColorizationTrainer(config)
    
    # Start training
    trainer.train(train_loader, val_loader, config['num_epochs'])

if __name__ == "__main__":
    main()