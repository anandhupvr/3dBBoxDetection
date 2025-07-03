import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import  torch.nn.functional as F
from PIL import Image
import numpy as np
from torch_geometric.data import Data, Batch

from utils import bbox3d_to_parametric
from torch.utils.data import random_split


class Challenge3DDataset(Dataset):
    def __init__(self, root_dir, augment=False, npoints=4096, img_size=(480, 640)):
        self.root_dir = root_dir
        self.img_size = img_size
        self.sample_dirs =  sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.npoints = npoints
        self.augment = augment

        self.rgb_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std =[0.229, 0.224, 0.225])
        ])

    def sample_pc(self, points, n_samples):
        if len(points) <= n_samples:
            return points
        
        # Sample 70% from lower half (ground plane)
        z_val = points[:,2]
        upper_idx = np.where(z_val < np.median(z_val))[0]
        lower_idx = np.where(z_val >= np.median(z_val))[0]
        
        n_lower = int(n_samples * 0.7)
        n_upper = n_samples - n_lower
        
        lower_sample = np.random.choice(lower_idx, n_lower, replace=len(lower_idx) < n_lower)
        upper_sample = np.random.choice(upper_idx, n_upper, replace=len(upper_idx) < n_upper)
        
        return points[np.concatenate([lower_sample, upper_sample])]

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        # rgb image
        image = Image.open(os.path.join(sample_dir, "rgb.jpg")).convert("RGB")
        org_w, org_h = image.size

        if self.rgb_transform:
            image = self.rgb_transform(image)

        scale_w = self.img_size[1] / org_w
        scale_h = self.img_size[0] / org_h

        # point cloud
        pc = np.load(os.path.join(sample_dir, "pc.npy"))
        pc[0] *= scale_w
        pc[1] *= scale_h 
        xyz = pc.reshape(3, -1).T
        pc = xyz[np.isfinite(xyz).all(1)]
        pc = torch.from_numpy(self.sample_pc(xyz, self.npoints)).float()

        pc_center = pc.mean(dim=0)
        pc -= pc_center
        max_dist = torch.max(torch.norm(pc, dim=1))
        pc /= max_dist


        assert torch.allclose(pc.mean(dim=0), torch.zeros(3), atol=1e-4)
        assert torch.max(torch.norm(pc, dim=1)) <= 1.0 + 1e-4

        #####  there are few flat plane , consider taking less points from there 

        # mask
        mask = np.load(os.path.join(sample_dir, "mask.npy"))
        # Create single-channel combined mask
        combined_mask = np.zeros(mask.shape[1:], dtype=np.uint8)  # [H, W]
        for obj_idx in range(mask.shape[0]):
            combined_mask[mask[obj_idx] > 0] = obj_idx + 1  # 0=bg, 1=obj1, ..., 18=obj18
        # TODO : Handle overlapping masks
        # Convert to tensor and resize
        combined_mask = torch.from_numpy(combined_mask).unsqueeze(0)  # [1, H, W]
        combined_mask = F.interpolate(
            combined_mask.float().unsqueeze(0),  # [1, 1, H, W] 
            size=(15, 20),
            mode='nearest'
        ).squeeze().long()  # [15, 20]

        #3dbbox
        bbox3d = np.load(os.path.join(sample_dir, "bbox3d.npy"))
        bbox3d_scaled = bbox3d.copy()
        bbox3d_scaled[:, :, 0] *= scale_w  # Scale X coordinates
        bbox3d_scaled[:, :, 1] *= scale_h  # Scale Y coordinates

        param_boxes = bbox3d_to_parametric(bbox3d_scaled) 
        bbox3d = torch.from_numpy(param_boxes).float()
        
        bbox3d[:, :3] -= pc_center
        bbox3d[:, :3] /= max_dist
        bbox3d[:, 3:6] /= max_dist

        if (self.augment):
            image, pc, combined_mask, bbox3d = self.apply_augmentation(image, pc, combined_mask, bbox3d)
        # bbox3d_scaled -= pc_center.numpy()
        # bbox3d_scaled /= (max_dist.item() + 1e-6)

        # bbox3d_centered = bbox3d_scaled - pc_center.reshape(1, 1, 3)
        # bbox3d = bbox3d - pc_center.reshape(1, 1, 3)


        # Add num_boxes to the return dictionary
        return {
            "image": image,            # [3, 480, 640]
            "pointcloud": pc,           # [1024, 3]
            "mask": combined_mask,      # [15, 20]
            "bbox3d": bbox3d,           # [N, 8, 3] (variable)
            "num_boxes": torch.tensor(bbox3d.shape[0])  # scalar
        }
    
    def apply_augmentation(self, image, pc, mask, bbox3d):
        
        if (torch.rand(1).item() < 0.5):
            image = torch.flip(image, [2])
            pc[:, 0] = -pc[:, 0]
            mask = torch.flip(mask, [1])
            if (len(bbox3d) > 0):
                bbox3d[:, :, 0] = -bbox3d[:, :, 0]
                bbox3d[:, 6] = -bbox3d[:, 6] # yaw

        if (torch.rand(1).item() < 0.5):
            angle = np.random.uniform(-10, 10) # recheck this part
            rad = np.deg2rad(angle)
            cos, sin = np.cos(rad), np.sin(rad)
            rot_mat = torch.tensor([
                [cos, 0, sin],
                [0, 1, 0],
                [-sin, 0, cos]
            ], dtype=torch.float32)
            
            pc = torch.mm(pc, rot_mat.T)
            if len(bbox3d) > 0:
                bbox3d[:, :3] = torch.mm(bbox3d[:, :3], rot_mat.T)
                bbox3d[:, 6] += rad  # yaw
        
        return image, pc, mask, bbox3d

def collate_fn(batch):
    # Handle fixed-size elements by stacking
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    num_boxes = torch.stack([item['num_boxes'] for item in batch])
    
    # Handle variable-sized bboxes - keep as list or pad them
    bboxes = [item['bbox3d'] for item in batch]

    data_list = []
    for i, item in enumerate(batch):
        pointcloud = item['pointcloud']  # [N, 3]
        data = Data(pos=pointcloud)
        data_list.append(data)
    
    pointcloud_batch = Batch.from_data_list(data_list)
    
    # Option 1: Return as list of tensors
    return {
        'images': images,          # [B, 3, 480, 640]
        'pointclouds': pointcloud_batch, # [B, 1024, 3]
        'masks': masks,            # [B, 15, 20]
        'bboxes': bboxes,          # List of [N_i, 8, 3] tensors
        'num_boxes': num_boxes     # [B]
    }

def get_splits(dataset_root, val_ratio=0.1, test_ratio=0.1):
    full_dataset = Challenge3DDataset(dataset_root, augment=False)
    
    # Calculate lengths
    total_len = len(full_dataset)
    val_len = int(total_len * val_ratio)
    test_len = int(total_len * test_ratio)
    train_len = total_len - val_len - test_len
    
    # Split the dataset
    train, val, test = random_split(
        full_dataset, 
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_dataset = Challenge3DDataset(
        root_dir=dataset_root,
        augment=True,  # Enable augmentation for training
        npoints=full_dataset.npoints,
        img_size=full_dataset.img_size
    )
    
    # Get the original indices from the split
    train_indices = train.indices
    
    # Create a Subset using the augmented dataset with original indices
    final_train = torch.utils.data.Subset(train_dataset, train_indices)
    
    return final_train, val, test