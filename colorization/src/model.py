# src/model.py - FIXED VERSION

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Double Convolution block used in U-Net
    (convolution => [BN] => ReLU) * 2
    """
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv - FIXED VERSION"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # Use bilinear upsampling or transpose conv
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle potential size differences - FIXED INDEXING
        diffY = x2.size()[2] - x1.size()[2]  # Fixed: was [11]
        diffX = x2.size()[3] - x1.size()[3]  # Fixed: was [12]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension  
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Final output convolution"""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class PeriodEmbedding(nn.Module):
    """
    Embedding layer for historical period information
    Helps model learn period-specific color characteristics
    """
    
    def __init__(self, num_periods=4, embedding_dim=64):
        super().__init__()
        self.period_dict = {
            'default': 0,
            '1920s': 1, 
            'wwii': 2,
            'victorian': 3
        }
        
        self.embedding = nn.Embedding(num_periods, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 256)  # Project to feature space
        
    def forward(self, period_labels, spatial_size):
        """
        Args:
            period_labels: List of period strings for batch
            spatial_size: (H, W) tuple for spatial dimensions
        Returns:
            period_features: [B, 256, H, W] tensor
        """
        batch_size = len(period_labels)
        
        # Convert period labels to indices
        period_indices = torch.tensor([
            self.period_dict.get(period, 0) for period in period_labels
        ], device=next(self.parameters()).device)
        
        # Get embeddings [B, embedding_dim]
        period_emb = self.embedding(period_indices)
        
        # Project to feature space [B, 256] 
        period_features = self.fc(period_emb)
        
        # Reshape to spatial dimensions [B, 256, H, W]
        H, W = spatial_size
        period_features = period_features.unsqueeze(-1).unsqueeze(-1)
        period_features = period_features.expand(-1, -1, H, W)
        
        return period_features

class HistoricalColorizationUNet(nn.Module):
    """
    U-Net architecture for historical image colorization
    Input: L channel (grayscale) [B, 1, H, W]
    Output: ab channels (color) [B, 2, H, W]
    """
    
    def __init__(self, n_channels=1, n_classes=2, bilinear=False, use_period_embedding=True):
        super(HistoricalColorizationUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_period_embedding = use_period_embedding
        
        # Period embedding for historical context
        if use_period_embedding:
            self.period_embedding = PeriodEmbedding()
        
        # Encoder path
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # Bottleneck with period conditioning - FIXED
        factor = 2 if bilinear else 1
        if use_period_embedding:
            # Input to down4 will be 512 + 256 = 768 channels
            self.down4 = Down(768, 1024 // factor)
        else:
            self.down4 = Down(512, 1024 // factor)
        
        # Decoder path  
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output layer
        self.outc = OutConv(64, n_classes)
        
        # Output activation - Tanh for ab channels in [-1, 1] range
        self.output_activation = nn.Tanh()

    def forward(self, x, period_labels=None):
        """
        Forward pass
        Args:
            x: Input L channel [B, 1, H, W]
            period_labels: List of period strings for each item in batch
        Returns:
            ab_pred: Predicted ab channels [B, 2, H, W]
        """
        # Encoder
        x1 = self.inc(x)          # [B, 64, H, W]
        x2 = self.down1(x1)       # [B, 128, H/2, W/2]
        x3 = self.down2(x2)       # [B, 256, H/4, W/4]
        x4 = self.down3(x3)       # [B, 512, H/8, W/8]
        
        # Add period conditioning at bottleneck - FIXED
        if self.use_period_embedding and period_labels is not None:
            spatial_size = (x4.size(2), x4.size(3))  # (H/8, W/8)
            period_features = self.period_embedding(period_labels, spatial_size)
            x4_conditioned = torch.cat([x4, period_features], dim=1)  # [B, 768, H/8, W/8]
        else:
            x4_conditioned = x4
        
        # Bottleneck
        x5 = self.down4(x4_conditioned)  # [B, 1024, H/16, W/16]
        
        # Decoder with skip connections
        x = self.up1(x5, x4)      # [B, 512, H/8, W/8]
        x = self.up2(x, x3)       # [B, 256, H/4, W/4]
        x = self.up3(x, x2)       # [B, 128, H/2, W/2]
        x = self.up4(x, x1)       # [B, 64, H, W]
        
        # Output ab channels
        ab_pred = self.outc(x)    # [B, 2, H, W]
        ab_pred = self.output_activation(ab_pred)  # Tanh activation
        
        return ab_pred

    def predict_full_image(self, L_channel, period='default'):
        """
        Predict colorization for a single image
        Args:
            L_channel: [1, H, W] or [H, W] tensor
            period: Historical period string
        Returns:
            colorized_rgb: [3, H, W] RGB image
        """
        self.eval()
        
        with torch.no_grad():
            # Ensure correct input shape [1, 1, H, W]
            if L_channel.dim() == 2:
                L_channel = L_channel.unsqueeze(0).unsqueeze(0)
            elif L_channel.dim() == 3:
                L_channel = L_channel.unsqueeze(0)
            
            # Predict ab channels
            ab_pred = self.forward(L_channel, period_labels=[period])
            
            # Denormalize predictions
            L_denorm = (L_channel.squeeze(0) + 1.0) * 50.0  # [1, H, W]
            ab_denorm = ab_pred.squeeze(0) * 128.0          # [2, H, W]
            
            # Combine LAB channels
            lab_combined = torch.cat([L_denorm, ab_denorm], dim=0)  # [3, H, W]
            
            # Convert LAB to RGB
            from src.utils import lab_to_rgb
            rgb_output = lab_to_rgb(lab_combined)
            
            return rgb_output

def create_model(device='cpu', use_period_embedding=True):
    """
    Create and initialize the colorization model
    """
    model = HistoricalColorizationUNet(
        n_channels=1,
        n_classes=2, 
        bilinear=True,
        use_period_embedding=use_period_embedding
    )
    
    model = model.to(device)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    return model

# Test function for model
if __name__ == "__main__":
    # Test model creation and forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(device)
    
    # Test input
    test_L = torch.randn(2, 1, 256, 256).to(device)
    test_periods = ['1920s', 'wwii']
    
    # Forward pass
    output = model(test_L, test_periods)
    print(f"Input shape: {test_L.shape}")
    print(f"Output shape: {output.shape}")
    print("Model test successful!")