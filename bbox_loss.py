import torch
import torch.nn as nn

class BoxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_boxes, gt_boxes, confidences, uncertainties):
        """
        Args:
            pred_boxes: [B,A,H,W,7]
            gt_boxes: List of tensors (variable length per sample)
            confidences: [B,A,H,W,1]
            uncertainties: [B,A,H,W,7]
        """
        # 1. Match predictions to GT (using mask IoU)
        matched_preds, matched_gts = self.match_boxes(pred_boxes, gt_boxes)
        
        # 2. Confidence loss (focal loss)
        conf_loss = self.focal_loss(confidences, matched_gts['iou_scores'])
        
        # 3. Box regression (uncertainty-weighted)
        box_loss = self.uncertainty_loss(
            matched_preds['params'],
            matched_gts['params'],
            matched_preds['uncertainties']
        )
        
        return conf_loss + 0.5 * box_loss

    def uncertainty_loss(self, pred, gt, sigma):
        """Heteroscedastic uncertainty weighting"""
        return torch.mean(0.5 * torch.exp(-sigma) * (pred - gt)**2 + 0.5 * sigma)