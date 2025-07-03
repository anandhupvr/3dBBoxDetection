import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedFusion(nn.Module):
    def __init__(self, rgb_channels=576, pc_channels=1024, out_channels=512):
        super().__init__()
        
        # RGB feature processing
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(rgb_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Point cloud feature processing
        self.pc_fc = nn.Sequential(
            nn.Linear(pc_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Cross-modality attention
        self.attention = nn.Sequential(
            nn.Conv2d(256 + 256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.out_conv = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(self, rgb_feats, pc_feats, mask):
        """
        Args:
            rgb_feats: [B, 576, 15, 20]
            pc_feats: [B, 1024]
            mask: [B, 480, 640] (downsample needed)
        Returns:
            fused_feats: [B, out_channels, 15, 20]
        """
        # Process RGB features
        rgb = self.rgb_conv(rgb_feats)  # [B, 256, 15, 20]
        
        # Process point cloud features
        pc = self.pc_fc(pc_feats)  # [B, 256]
        pc = pc.unsqueeze(-1).unsqueeze(-1)  # [B, 256, 1, 1]
        pc = pc.expand(-1, -1, 15, 20)  # [B, 256, 15, 20]

        mask = mask.unsqueeze(1)
        # Concatenate features
        combined = torch.cat([rgb, pc], dim=1)  # [B, 512, 15, 20]
        
        # Compute attention weights
        attn = self.attention(combined)  # [B, 1, 15, 20]
        
        # Apply mask-guided attention
        binary_mask = (mask > 0).float()
        attn = attn * binary_mask 
        
        # Fuse features
        fused = rgb * attn + pc * (1 - attn)
        
        # Final projection
        return self.out_conv(fused)  # [B, 512, 15, 20]