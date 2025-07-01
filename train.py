import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torch.utils.data import DataLoader
from data_loader import Challenge3DDataset, collate_fn

from models.model import Simple3DDetectionModel

from utils import BoxLoss, Mask3DLoss

import open3d as o3d
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_sample(sample, idx=0):

    # Extract data
    image = sample['images'][idx].permute(1, 2, 0).cpu().numpy()
    pointcloud = sample['pointclouds'][idx].pos.cpu().numpy()
    mask = sample['masks'][idx].cpu().numpy()
    bboxes = sample['bboxes'][idx].cpu().numpy() if isinstance(sample['bboxes'][idx], torch.Tensor) else sample['bboxes'][idx]
    
    # Normalize image if needed for display
    image_disp = (image - image.min()) / (image.max() - image.min())
    
    # Visualization 1: RGB Image with Mask Overlay
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_disp)
    plt.title("RGB Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title("Segmentation Mask")
    plt.show()
    # import pdb; pdb.set_trace()
    # Visualization 2: 3D Point Cloud with BBoxes
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud.astype(np.float64))
    
    # Create bbox geometries
    bbox_geoms = []
    for box in bboxes:
        box = box.astype(np.float64)  # Ensure correct type
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(box)
        )
        bbox.color = [1, 0, 0]  # Red
        bbox_geoms.append(bbox)
    
    # Visualize
    o3d.visualization.draw_geometries([pcd, *bbox_geoms], 
                                     window_name="3D Point Cloud with BBoxes")

def visualize_batch(batch):
    visualize_sample(batch, idx=0)

def check_dataset(dataset_path="./dl_challenge"):
    dataset = Challenge3DDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    print(f"Dataset contains {len(dataset)} samples")
    
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i+1}")
        print(f"Images shape: {batch['images'].shape}")
        print(f"Pointcloud shape: {batch['pointclouds'].num_nodes}")
        print(f"Masks shape: {batch['masks'].shape}")
        print(f"Number of boxes: {batch['num_boxes']}")
        
        visualize_batch(batch)
        
        if i == 2:  # Just check first 3 batches
            break


def train_single_phase(model, dataloader, loss_fn, epochs=10, device='cuda'):
    # Initialize
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    writer = SummaryWriter()  # TensorBoard logging
    
    # MobileNet specific - fine-tuning setup
    for name, param in model.rgb_backbone.mobilenet.named_parameters():
        if 'features.0.' in name or 'features.1.' in name:  # First few layers
            param.requires_grad = False  # Keep early layers frozen
        
    best_loss = float('inf')
    
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
                # Forward pass
                optimizer.zero_grad()
                pred = model(batch)
                # import pdb; pdb.set_trace()
                # loss = loss_fn(pred, batch)
                loss = loss_fn(pred['pred_boxes'], pred['pred_scores'], batch['bboxes'])
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Logging
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        # Epoch handling
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch {epoch+1} - Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize components
    dataset = Challenge3DDataset("./dl_challenge")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, 
                           collate_fn=collate_fn, num_workers=4)
    
    model = Simple3DDetectionModel()
    # loss_fn = BoxLoss()
    loss_fn = Mask3DLoss()
    
    # Single-phase training
    train_single_phase(model, dataloader, loss_fn, epochs=50, device=device)

if __name__ == "__main__":
    main()
    # check_dataset()