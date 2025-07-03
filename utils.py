import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial import ConvexHull

import numpy as np
from sklearn.decomposition import PCA


def bbox3d_to_parametric(bboxes):
    """
    Convert 8-corner 3D boxes to parametric form:
    [center_x, center_y, center_z, dx, dy, dz, yaw]
    Args:
        bboxes: numpy array of shape [N, 8, 3]
    Returns:
        param_boxes: numpy array of shape [N, 7]
    """
    param_boxes = []

    for box in bboxes:
        # 1. Center
        center = box.mean(axis=0)

        # 2. Size (dx, dy, dz)
        dx = np.linalg.norm(box[0] - box[1])
        dy = np.linalg.norm(box[1] - box[2])
        dz = np.linalg.norm(box[0] - box[4])  # vertical

        # 3. Estimate yaw using PCA on XY plane
        xy = box[:, :2]  # [8, 2]
        pca = PCA(n_components=2)
        pca.fit(xy)
        direction = pca.components_[0]
        yaw = np.arctan2(direction[1], direction[0])

        param_box = np.array([center[0], center[1], center[2], dx, dy, dz, yaw])
        param_boxes.append(param_box)

    return np.array(param_boxes)

def parametric_to_bbox3d(box):
    # Ensure all inputs are float64
    cx, cy, cz, dx, dy, dz, yaw = map(float, box)
    
    # Create unrotated corners (local coordinates)
    half_dx = dx / 2
    half_dy = dy / 2
    half_dz = dz / 2
    
    corners = np.array([
        [ half_dx,  half_dy,  half_dz],
        [ half_dx, -half_dy,  half_dz],
        [-half_dx, -half_dy,  half_dz],
        [-half_dx,  half_dy,  half_dz],
        [ half_dx,  half_dy, -half_dz],
        [ half_dx, -half_dy, -half_dz],
        [-half_dx, -half_dy, -half_dz],
        [-half_dx,  half_dy, -half_dz]
    ], dtype=np.float64)  # Explicitly set dtype
    
    # Rotation matrix (Z-axis)
    rot_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ], dtype=np.float64)
    
    rotated_corners = np.dot(corners, rot_matrix.T)
    rotated_corners += np.array([cx, cy, cz], dtype=np.float64)  # Ensure float64
    
    return rotated_corners


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


#  not the best 
def calculate_aabb_iou(box1, box2):
    
    box1_corners = parametric_to_bbox3d(box1)
    box2_corners = parametric_to_bbox3d(box2)


    # Compute the axis-aligned bounding box for each box
    min1 = np.min(box1_corners, axis=0)
    max1 = np.max(box1_corners, axis=0)
    min2 = np.min(box2_corners, axis=0)
    max2 = np.max(box2_corners, axis=0)

    # Compute the intersection box
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)

    # Check if there is an intersection
    if np.all(inter_min <= inter_max):
        intersection_volume = np.prod(inter_max - inter_min)
    else:
        intersection_volume = 0.0

    # Compute the volumes of both boxes
    vol1 = np.prod(max1 - min1)
    vol2 = np.prod(max2 - min2)

    # Compute the union volume
    union_volume = vol1 + vol2 - intersection_volume

    # Return IoU
    return intersection_volume / union_volume if union_volume > 1e-6 else 0.0

class BoxLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reg_loss = nn.SmoothL1Loss(reduction='none')
        
    def forward(self, preds, batch):
        """poly_area
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
            box_params = torch.stack([box for box in boxes])

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
        self.reg_loss = nn.SmoothL1Loss(beta=0.1, reduction='none')
        self.conf_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.eps = 1e-6

    def forward(self, pred_boxes, pred_scores, gt_boxes):
        total_loss = 0.0
        batch_size = len(pred_boxes)
        
        for boxes, scores, gt in zip(pred_boxes, pred_scores, gt_boxes):
            # Input validation
            if torch.isnan(boxes).any() or torch.isnan(scores).any():
                print("NaN detected in predictions!")
                continue
                
            if len(gt) > 0 and torch.isnan(gt).any():
                print("NaN detected in ground truth!")
                continue

            # Convert GT to tensor 
            gt_params = torch.stack([b for b in gt]) if len(gt) > 0 else torch.zeros(0, 7, device=boxes.device)
            
            # Case 1: No objects
            if len(gt_params) == 0:
                # Clamp scores before squaring to prevent explosion
                safe_scores = torch.clamp(scores, -10, 10)
                total_loss += torch.mean(safe_scores**2)
                continue
   
            # Case 2/3/4 unified handling
            num_pairs = min(len(boxes), len(gt_params))
            # import pdb; pdb.set_trace()
            # Regression loss for matched pairs
            if num_pairs > 0:
                reg_loss = self.reg_loss(boxes[:num_pairs], gt_params[:num_pairs])
                reg_loss = torch.mean(reg_loss)  # Take mean over all elements
                total_loss += reg_loss
                
                # Classification loss for matched pairs
                conf_loss = self.conf_loss(scores[:num_pairs], 
                                         torch.ones_like(scores[:num_pairs]))
                conf_loss = torch.mean(conf_loss)
                total_loss += conf_loss

            # False positive penalty (extra predictions)
            if len(boxes) > len(gt_params):
                extra_scores = scores[len(gt_params):]
                safe_scores = torch.clamp(extra_scores, -10, 10)
                total_loss += 0.3 * torch.mean(safe_scores**2)  # Mean instead of sum

            # False negative penalty (missed ground truths)
            if len(boxes) < len(gt_params):
                total_loss += 0.3 * (len(gt_params) - len(boxes)) / len(gt_params)  # Normalized

        # Normalize by batch size (with epsilon to prevent div by zero)
        return total_loss / (batch_size + self.eps)
    