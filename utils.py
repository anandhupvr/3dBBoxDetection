import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F


def corners_to_parametric(corners):
    device = corners.device

    # Center calculation (same)
    center = corners.mean(dim=0)

    # Size calculation using PCA for orientation-agnostic dimensions
    cov = (corners - center).T @ (corners - center)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    size = 2 * torch.sqrt(eigvals)[[1,2,0]]  # Reorder to w,h,d

    # Yaw calculation using dominant direction
    front_vector = eigvecs[:,0]  # Principal component
    yaw = torch.atan2(front_vector[1], front_vector[0])

    return torch.cat([center, size, yaw.unsqueeze(0)]).to(device)


def decode_predictions(preds, conf_thresh=0.5):
    bbox_params, confidences, _ = preds  # [B,A,H,W,...]
    
    # Convert to global coordinates
    grid_x = torch.arange(0, 20).view(1,1,-1,1).to(bbox_params.device)
    grid_y = torch.arange(0, 15).view(1,-1,1,1).to(bbox_params.device)
    
    # Decode box parameters
    boxes = torch.zeros_like(bbox_params)
    boxes[..., 0] = (bbox_params[..., 0] + grid_x) * (640/20)  # x
    boxes[..., 1] = (bbox_params[..., 1] + grid_y) * (480/15)  # y 
    boxes[..., 2] = bbox_params[..., 2]                        # z
    boxes[..., 3:6] = torch.exp(bbox_params[..., 3:6])         # w,h,l
    boxes[..., 6] = bbox_params[..., 6]                       # θ
    
    # Filter by confidence
    mask = confidences > conf_thresh
    return boxes[mask], confidences[mask]


def nms_3d(boxes, scores, iou_threshold=0.5):
    """
    3D Non-Maximum Suppression
    Args:
        boxes: [N,8,3] corner coordinates
        scores: [N] confidence scores
    Returns:
        keep_indices: indices of kept boxes
    """
    # Calculate 3D IoU matrix
    iou_matrix = calculate_3d_iou_open3d(boxes, boxes)  # [N,N]
    
    # Standard NMS implementation
    keep = []
    order = scores.argsort(descending=True)
    
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        
        # Calculate IoU with remaining boxes
        ious = iou_matrix[i, order[1:]]
        
        # Keep boxes with IoU < threshold
        keep_mask = ious < iou_threshold
        order = order[1:][keep_mask]
        
    return torch.tensor(keep, device=boxes.device)


def calculate_3d_iou_open3d(box1, box2):
    """Calculate exact 3D IoU using Open3D"""
    # Create oriented bounding boxes
    obb1 = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(box1))
    obb2 = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(box2))
    
    # Compute intersection volume
    intersection = o3d.geometry.OrientedBoundingBox.get_intersection_volume(obb1, obb2)
    
    # Compute volumes
    vol1 = obb1.volume()
    vol2 = obb2.volume()
    union = vol1 + vol2 - intersection
    
    return intersection / union if union > 1e-6 else 0.0

class BoxLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reg_loss = nn.SmoothL1Loss(reduction='none')
        
    def forward(self, preds, batch):
        """
        Args:
            preds: Tuple of (bbox_params, confidences, uncertainties)
                   - bbox_params: [B,A,H,W,7]
                   - confidences: [B,A,H,W,1]
                   - uncertainties: [B,A,H,W,7]
            batch: Dictionary from dataloader
        """
        bbox_pred, conf_pred, uncert_pred = preds
        B, A, H, W, _ = bbox_pred.shape
        device = bbox_pred.device

        # Initialize target tensors
        gt_params = torch.zeros((B, A, H, W, 7), device=device)
        gt_conf = torch.zeros((B, A, H, W, 1), device=device)
        gt_mask = torch.zeros((B, A, H, W), device=device, dtype=torch.bool)

        # 1. Prepare targets for each sample
        for b in range(B):
            mask = batch['masks'][b]  # [H,W]
            boxes = batch['bboxes'][b]  # [N,8,3]

            if len(boxes) == 0:
                continue  # No boxes in this sample

            # Convert boxes to parametric format
            box_params = torch.stack([corners_to_parametric(box) for box in boxes])

            # Find mask regions for each object
            for obj_idx in range(len(boxes)):
                obj_mask = (mask == (obj_idx+1))  # +1 since mask 0=background

                if obj_mask.any():
                    import pdb; pdb.set_trace()
                    # Assign to all anchors in masked region
                    gt_params[b, :, obj_mask] = box_params[obj_idx]
                    gt_conf[b, :, obj_mask] = 1.0
                    gt_mask[b, :, obj_mask] = True

        # 2. Confidence loss (Focal Loss)
        conf_loss = self.focal_loss(conf_pred, gt_conf)

        # 3. Regression loss (only on positive locations)
        reg_loss = self.reg_loss(bbox_pred, gt_params)
        reg_loss = (torch.exp(-uncert_pred) * reg_loss + 0.5 * uncert_pred)
        reg_loss = reg_loss[gt_mask].mean()  # Only compute where we have objects
        
        return conf_loss + reg_loss
    
    def focal_loss(self, pred, target):
        pt = torch.where(target == 1, pred, 1-pred)
        focal_weight = (1-pt).pow(self.gamma)
        loss = F.binary_cross_entropy(pred, target, reduction='none')
        return (self.alpha * focal_weight * loss).mean()

class Mask3DLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.reg_loss = nn.SmoothL1Loss()
        self.conf_loss = nn.BCELoss()

    def forward(self, pred_boxes, pred_scores, gt_boxes):

        total_loss = 0.0

        for boxes, scores, gt in zip(pred_boxes, pred_scores, gt_boxes):
            # import pdb; pdb.set_trace()
            gt = gt.to(boxes.device)
            # Convert GT corners to parametric
            # gt_params = torch.stack([corners_to_parametric(b.to(boxes.device)) for b in gt]) if len(gt) > 0 else torch.zeros(0,7, device=boxes.device)
            gt_params = torch.stack([corners_to_parametric(b.to(boxes.device)).to(boxes.device) for b in gt]) if len(gt) > 0 else torch.zeros(0, 7, device=boxes.device)
            gt_params = gt_params.to(boxes.device)

            # Case 1: No objects
            if len(gt) == 0:
                total_loss += torch.sum(scores**2)  # Penalize FP
                continue
   
            # Case 2: Perfect match
            if len(boxes) == len(gt_params):
                total_loss += self.reg_loss(boxes, gt_params)
                total_loss += self.conf_loss(scores, torch.ones_like(scores))

            # Case 3: More predictions than GT
            elif len(boxes) > len(gt_params):
                total_loss += self.reg_loss(boxes[:len(gt_params)], gt_params)
                total_loss += self.conf_loss(scores[:len(gt_params)], torch.ones_like(scores[:len(gt_params)]))
                total_loss += 0.3 * torch.sum(scores[len(gt_params):]**2)  # FP penalty

            # Case 4: Fewer predictions than GT
            else:
                total_loss += self.reg_loss(boxes, gt_params[:len(boxes)])
                total_loss += self.conf_loss(scores, torch.ones_like(scores))
                total_loss += 0.3 * (len(gt_params) - len(boxes))  # FN penalty

        return total_loss / len(pred_boxes)