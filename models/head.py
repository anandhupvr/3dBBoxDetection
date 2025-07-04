import torch
import torch.nn as nn
import torch.nn.functional as F   

class MaskBBoxPredictor(nn.Module):
    def __init__(self, feat_channels=512):
        super().__init__()
        
        self.coord_conv = nn.Sequential(
            nn.Conv2d(2, 64, 1),
            nn.ReLU()
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(feat_channels + 64, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(4, 128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(4, 64),
            nn.ReLU()
        )

        # Prediction heads
        self.bbox_head = nn.Conv2d(64, 7, 1) # x, y, z, w, h , l, theta

        self.conf_head = nn.Sequential(
            nn.Conv2d(64, 1, 1)
        )

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, masks):

        B, _, H, W = x.shape
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))

        batch_boxes, batch_scores = [], []
        grid = torch.stack([grid_x, grid_y], dim=0).float().to(x.device)
        grid = grid.unsqueeze(0).expand(B,-1,-1,-1)  # [B,2,H,W]
        grid_feats = self.coord_conv(grid)

        x = torch.cat([x, grid_feats], dim=1)
        x = self.encoder(x)

        for b in range(x.shape[0]):
            obj_ids = torch.unique(masks[b])[1:]
            sample_boxes, sample_scores = [], []

            for obj_id in obj_ids:
                # 1.  Get object mask
                obj_mask = (masks[b] == obj_id).float()

                # 2. Mask feature and average pool
                masked_feats = x[b] * obj_mask.unsqueeze(0) # 65, 15, 20
                # pooled = masked_feats.sum(dim=(1,2)) / (obj_mask.sum() + 1e-6) # 64
                mask_sum = obj_mask.sum()
                if mask_sum < 1e-4:
                    continue  # skip bad mask

                pooled = masked_feats.sum(dim=(1,2)) / mask_sum
                pooled = pooled.view(1, 64, 1, 1)

                # 3. Predict 
                box = self.bbox_head(pooled).squeeze()
                score = self.conf_head(pooled).squeeze()

                sample_boxes.append(box)
                sample_scores.append(score)

            # Handle empty case
            boxes = torch.stack(sample_boxes) if sample_boxes else torch.zeros(0, 7, device=x.device)
            scores = torch.stack(sample_scores) if sample_scores else torch.zeros(0, device=x.device)

            batch_boxes.append(boxes)
            batch_scores.append(scores)

        return batch_boxes, batch_scores