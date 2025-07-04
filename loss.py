import torch
import torch.nn as nn

from utils import hungarian_matching
    
class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.conf_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.center_loss = nn.SmoothL1Loss(reduction='none')
        self.dims_loss = nn.SmoothL1Loss(reduction='none')

        self.w_conf = 1
        self.w_bbox = 2
        self.w_angle = 2

    def forward(self, pred_boxes, pred_scores, gt_boxes):
        total_loss = 0.0
        total_boxes = 0

        for batch_idx in range(len(gt_boxes)):
            boxes = pred_boxes[batch_idx]
            scores = pred_scores[batch_idx]
            gt = torch.stack([b for b in gt_boxes[batch_idx]]) if len(gt_boxes[batch_idx]) > 0 else torch.zeros(0, 7, device=boxes.device)

            if len(gt) == 0:
                conf_loss = self.conf_loss(boxes, torch.zeros_like(boxes, device=boxes.device))
                total_loss += conf_loss # weights ? + box weights
                continue
            
            if (len(boxes) == 0):
                total_loss += self.w_conf + torch.tensor(1.0, device=boxes.device)
                continue

            # Hungarian matchin
            # import pdb; pdb.set_trace()

            matches, unmatched_preds, unmatched_gts = hungarian_matching(boxes, gt, scores)

            # objectness / classification 
            conf_targets = torch.zeros_like(scores)
            if (len(matches) > 0):
                matched_pred_indices = [m[0] for m in matches]
                conf_targets[matched_pred_indices] = 1.0

            conf_loss = self.conf_loss(scores, conf_targets)

            # regression loss
            reg_loss = torch.tensor(0.0, device=boxes.device)
            angle_loss = torch.tensor(0.0, device=boxes.device)

            if len(matches) > 0:
                matched_pred_boxes = boxes[[m[0] for m in matches]]
                matched_gt_boxes = gt[[m[1] for m in matches]]

                # Center loss
                center_loss = self.center_loss(
                    matched_pred_boxes[:, :3], 
                    matched_gt_boxes[:, :3]
                ).sum(dim=1).mean()
            
                dims_loss = self.dims_loss(
                    matched_pred_boxes[:, 3:6], 
                    matched_gt_boxes[:, 3:6]
                ).sum(dim=1).mean()

                angle_diff = matched_pred_boxes[:, 6] - matched_gt_boxes[:, 6]
                angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
                angle_loss = (angle_diff ** 2).mean()
                
                reg_loss = center_loss + dims_loss + self.w_angle * angle_loss

            batch_loss = (
                self.w_conf * conf_loss.mean() +
                self.w_bbox * reg_loss
            ) # probably not a proper weighting

            # unmatched predictions

            if (len(unmatched_preds) > 0):
                batch_loss += self.w_conf * conf_loss[unmatched_preds].mean()
                
            if len(unmatched_gts) > 0:
                batch_loss += self.w_conf * torch.tensor(1.0, device=boxes.device)
            
            total_loss += batch_loss
            total_boxes += len(gt)

        return total_loss / max(1, total_boxes)
