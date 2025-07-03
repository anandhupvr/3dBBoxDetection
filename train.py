import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torch.utils.data import DataLoader
from data_loader import Challenge3DDataset, collate_fn, denormalize_boxes

from models.model import Simple3DDetectionModel

from utils import BoxLoss, Mask3DLoss, calculate_aabb_iou

import open3d as o3d
import matplotlib
matplotlib.use('qtagg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from torch.utils.data import random_split

def denormalize_image(image):
    """Reverse ImageNet normalization for visualization"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean  # Denormalize
    return np.clip(image, 0, 1)

def visualize_sample(sample, idx=0):
    # import pdb; pdb.set_trace()

    # # Extract data
    image = sample['images'][idx].permute(1, 2, 0).cpu().numpy()
    pointcloud = sample['pointclouds'][idx].pos.cpu().numpy()
    mask = sample['masks'][idx].cpu().numpy()
    bboxes = sample['bboxes'][idx].cpu().numpy() if isinstance(sample['bboxes'][idx], torch.Tensor) else sample['bboxes'][idx]
    
    # # Normalize image if needed for display
    image_disp = denormalize_image(image)
    
    # # Visualization 1: RGB Image with Mask Overlay
    plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    plt.imshow(image_disp)
    plt.title("RGB Image")
    plt.show()
    

def visualize_batch(batch):
    batch_size = batch['images'].shape[0]

    for i in range(batch_size):
        visualize_sample(batch, idx=i)

def check_dataset(dataset_path="./dl_challenge"):
    dataset = Challenge3DDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    print(f"Dataset contains {len(dataset)} samples")
    
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i+1}")
        print(f"Images shape: {batch['images'].shape}")
        print(f"Pointcloud shape: {batch['pointclouds'].num_nodes}")
        print(f"Masks shape: {batch['masks'].shape}")
        print(f"Number of boxes: {len(batch['bboxes'])}")
        visualize_batch(batch)
        
        if i == 2:  # Just check first 3 batches
            break


def get_splits(dataset_root, val_ratio=0.1, test_ratio=0.1):
    full_dataset = Challenge3DDataset(dataset_root)
    
    # Calculate lengths
    val_len = int(len(full_dataset) * val_ratio)
    test_len = int(len(full_dataset) * test_ratio)
    train_len = len(full_dataset) - val_len - test_len
    
    # Split
    train, val, test = random_split(
        full_dataset, 
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )
    return train, val, test


def evaluate(model, test_loader, loss_fn, device, writer=None, epoch=0):
    model.eval()
    test_loss = 0.0

    iou_3d_list = []
    angle_diff_list = []
    center_dist_list = []
    size_diff_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            batch = {
                'images': batch['images'].to(device),
                'pointclouds': batch['pointclouds'].to(device),
                'masks':batch['masks'].to(device),
                'bboxes': [b.to(device) for b in batch['bboxes']],
                'num_boxes': batch['num_boxes'].to(device),
                'norm_params': batch['norm_params'].to(device)
            }
            pred = model(batch)
            loss = loss_fn(pred['pred_boxes'], pred['pred_scores'], batch['bboxes'])
            test_loss += loss.item()

            # Denormalize predictions and ground truth
            for i in range(len(batch['bboxes'])):
                denorm_pred_boxes = denormalize_boxes(
                    pred['pred_boxes'][i],
                    batch['norm_params'][i]
                )
                denorm_gt_boxes = denormalize_boxes(
                    batch['bboxes'][i],
                    batch['norm_params'][i]
                )

                for pred_box, gt_box in zip(denorm_pred_boxes, denorm_gt_boxes):
                    iou_3d = torch.tensor(calculate_aabb_iou(pred_box, gt_box), dtype=torch.float32, device=device)
                    angle_diff = torch.abs(pred_box[6] - gt_box[6])
                    center_dist = torch.norm(pred_box[:3] - gt_box[:3])
                    size_diff = torch.norm(pred_box[3:6] - gt_box[3:6])

                    iou_3d_list.append(iou_3d)
                    angle_diff_list.append(angle_diff)
                    center_dist_list.append(center_dist)
                    size_diff_list.append(size_diff)

    # Calculate average metrics
    avg_test_loss = test_loss / len(test_loader)
    avg_iou = torch.mean(torch.stack(iou_3d_list)).item()
    avg_angle_diff = torch.mean(torch.stack(angle_diff_list)).item()
    avg_center_dist = torch.mean(torch.stack(center_dist_list)).item()
    avg_size_diff = torch.mean(torch.stack(size_diff_list)).item()
    
    # Log to TensorBoard if writer provided
    if writer is not None:
        writer.add_scalar('Loss/test', avg_test_loss, epoch)
        writer.add_scalar('Metrics/3D_IoU', avg_iou, epoch)
        writer.add_scalar('Metrics/Angle_Diff', avg_angle_diff, epoch)
        writer.add_scalar('Metrics/Center_Dist', avg_center_dist, epoch)
        writer.add_scalar('Metrics/Size_Diff', avg_size_diff, epoch)
    
    print(f"\nTest Results - Loss: {avg_test_loss:.4f}")
    print(f"3D IoU: {avg_iou:.3f} | Angle Diff: {avg_angle_diff:.3f} rad")
    print(f"Center Dist: {avg_center_dist:.3f} m | Size Diff: {avg_size_diff:.3f} m")
    
    return {
        'test_loss': avg_test_loss,
        '3d_iou': avg_iou,
        'angle_diff': avg_angle_diff,
        'center_dist': avg_center_dist,
        'size_diff': avg_size_diff
    }


def validate(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = {
                    'images': batch['images'].to(device),
                    'pointclouds': batch['pointclouds'].to(device), 
                    'masks': batch['masks'].to(device),
                    'bboxes': [b.to(device) for b in batch['bboxes']],  # List of tensors
                    'num_boxes': batch['num_boxes'].to(device)
                    }
            pred = model(batch)
            loss = loss_fn(pred['pred_boxes'], pred['pred_scores'], batch['bboxes'])
            val_loss += loss.item()
    return val_loss / len(val_loader)

def train_single_phase(model, dataloader, val_loader, test_loader, loss_fn, epochs=10, device='cuda'):
    # Initialize
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    writer = SummaryWriter()  #  logging
    
    # MobileNet specific - fine-tuning setup
    for name, param in model.rgb_backbone.mobilenet.named_parameters():
        if 'features.0.' in name or 'features.1.' in name:  # First few layers
            param.requires_grad = False  # Keep early layers frozen
        
    best_loss = float('inf')
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        with tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
            for batch in pbar:
                # Prepare batch
                batch = {
                    'images': batch['images'].to(device),
                    'pointclouds': batch['pointclouds'].to(device), 
                    'masks': batch['masks'].to(device),
                    'bboxes': [b.to(device) for b in batch['bboxes']],  # List of tensors
                    'num_boxes': batch['num_boxes'].to(device)
                }

                def check_finite(tensor):
                    return torch.isfinite(tensor).all()
                assert check_finite(batch['images'])
                assert check_finite(batch['pointclouds'].pos)

                # Forward pass
                optimizer.zero_grad()
                pred = model(batch)
                # loss = loss_fn(pred, batch)
                loss = loss_fn(pred['pred_boxes'], pred['pred_scores'], batch['bboxes'])
                if torch.isnan(loss):
                    print(f"NaN detected in batch {batch}")
                    break

                if (torch.isfinite(loss)) : loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar('Loss/train', avg_loss, epoch)

        if (epoch % 2 == 0):
            val_loss = validate(model, val_loader, loss_fn, device)
            writer.add_scalar("Loss/val", val_loss, epoch)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                if (epoch > 25):
                    torch.save(model.state_dict(), f'best_model-epoch-{epoch}.pth')

        if ( epoch % 5  == 0):
            test_result = evaluate(model, test_loader, loss_fn, device, writer, epoch)

        if (epoch == epoch - 1):
            final_eval_result = evaluate(model, test_loader, loss_fn, device, writer, epoch)
            torch.save(model.state_dict(), "final_model.pth")
        
        print(f'Epoch {epoch+1} - Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize 
    # dataset = Challenge3DDataset("./dl_challenge")
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True, 
    #                        collate_fn=collate_fn, num_workers=4)

    train_set, val_set, test_set = get_splits("./dl_challenge")
    
    # Create dataloaders
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, 
                            collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=4, 
                          collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=4,
                           collate_fn=collate_fn, num_workers=2)

    
    model = Simple3DDetectionModel()
    # loss_fn = BoxLoss()
    loss_fn = Mask3DLoss()
    
    # Single-phase training
    train_single_phase(model, train_loader, val_loader, test_loader, loss_fn, epochs=50, device=device)

if __name__ == "__main__":
    main()
    # check_dataset()