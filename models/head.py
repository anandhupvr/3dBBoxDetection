import torch
import torch.nn as nn
import torch.nn.functional as F

class BBoxPredictor(nn.Module):
    def __init__(self, in_channels=512, num_anchors=1):
        super().__init__()
        # Shared convolutional layers
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Prediction branches
        self.bbox_pred = nn.Conv2d(128, 7 * num_anchors, kernel_size=1)  # 7 params per box
        self.conf_pred = nn.Conv2d(128, 1 * num_anchors, kernel_size=1)   # Confidence score
        self.uncertainty_pred = nn.Conv2d(128, 7 * num_anchors, kernel_size=1)  # Uncertainty per param
        
        # Initialize layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Fused features [B, C, H, W]
        Returns:
            bbox_params: [B, num_anchors, H, W, 7] (x,y,z,w,h,l,θ)
            confidences: [B, num_anchors, H, W, 1]
            uncertainties: [B, num_anchors, H, W, 7]
        """
        shared = self.shared_conv(x)
        
        # Predictions
        bbox_params = self.bbox_pred(shared)  # [B, 7*A, H, W]
        confidences = torch.sigmoid(self.conf_pred(shared))  # [B, 1*A, H, W]
        uncertainties = F.softplus(self.uncertainty_pred(shared))  # [B, 7*A, H, W]
        
        # Reshape outputs
        B, _, H, W = bbox_params.shape
        bbox_params = bbox_params.view(B, -1, 7, H, W).permute(0,1,3,4,2)  # [B,A,H,W,7]
        confidences = confidences.view(B, -1, 1, H, W).permute(0,1,3,4,2)   # [B,A,H,W,1]
        uncertainties = uncertainties.view(B, -1, 7, H, W).permute(0,1,3,4,2) # [B,A,H,W,7]
        
        return bbox_params, confidences, uncertainties