import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import  torch.nn.functional as F
from PIL import Image
import numpy as np
from torch_geometric.data import Data, Batch


class Challenge3DDataset(Dataset):
    def __init__(self, root_dir, npoints=4096, img_size=(480, 640)):
        self.root_dir = root_dir
        self.img_size = img_size
        self.sample_dirs =  sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.npoints = 1024

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
        lower_idx = np.where(z_val < np.median(z_val))[0]
        upper_idx = np.where(z_val >= np.median(z_val))[0]
        
        n_lower = int(n_samples * 0.7)
        n_upper = n_samples - n_lower
        
        lower_sample = np.random.choice(lower_idx, n_lower, replace=len(lower_idx) < n_lower)
        upper_sample = np.random.choice(upper_idx, n_upper, replace=len(upper_idx) < n_upper)
        
        return points[np.concatenate([lower_sample, upper_sample])]

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        #rgb image
        image = Image.open(os.path.join(sample_dir, "rgb.jpg")).convert("RGB")
        org_w, org_h = image.size

        if self.rgb_transform:
            image = self.rgb_transform(image)

        scale_w = self.img_size[1] / org_w  # width is 640
        scale_h = self.img_size[0] / org_h  # height is 480

        #point cloud
        pc = np.load(os.path.join(sample_dir, "pc.npy"))
        pc[0] *= scale_w
        pc[1] *= scale_h 
        xyz = pc.reshape(3, -1).T
        # pc = xyz[np.isfinite(xyz).all(1)]
        pc = torch.from_numpy(self.sample_pc(xyz, self.npoints)).float()
        pc = pc - pc.mean(dim=0) 
        #####  there is flat ground plane better to sample more from there ?

        #mask
        mask = np.load(os.path.join(sample_dir, "mask.npy"))
        # Create single-channel combined mask
        combined_mask = np.zeros(mask.shape[1:], dtype=np.uint8)  # [H, W]
        for obj_idx in range(mask.shape[0]):
            combined_mask[mask[obj_idx] > 0] = obj_idx + 1  # 0=bg, 1=obj1, ..., 18=obj18
        
        # Convert to tensor and resize
        combined_mask = torch.from_numpy(combined_mask).unsqueeze(0)  # [1, H, W]
        combined_mask = F.interpolate(
            combined_mask.float().unsqueeze(0),  # [1, 1, H, W] 
            size=(480, 640),
            mode='nearest'
        ).squeeze().long()  # [480, 640]

        #3dbbox
        bbox3d = np.load(os.path.join(sample_dir, "bbox3d.npy"))
        bbox3d = torch.from_numpy(bbox3d).float()
        
        # Add num_boxes to the return dictionary
        return {
            "image": image,            # [3, 480, 640]
            "pointcloud": pc,           # [1024, 3]
            "mask": combined_mask,      # [480, 640]
            "bbox3d": bbox3d,           # [N, 8, 3] (variable)
            "num_boxes": torch.tensor(bbox3d.shape[0])  # scalar
        }

def collate_fn(batch):
    # Handle fixed-size elements by stacking
    images = torch.stack([item['image'] for item in batch])
    # pointclouds = torch.stack([item['pointcloud'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    num_boxes = torch.stack([item['num_boxes'] for item in batch])
    
    # Handle variable-sized bboxes - keep as list or pad them
    bboxes = [item['bbox3d'] for item in batch]

    # Create PyG Data list
    data_list = []
    for i, item in enumerate(batch):
        pointcloud = item['pointcloud']  # [N, 3]
        data = Data(pos=pointcloud)  # add x=item['features'] if needed
        data_list.append(data)
    
    pointcloud_batch = Batch.from_data_list(data_list)
    
    # Option 1: Return as list of tensors
    return {
        'images': images,          # [B, 3, 480, 640]
        'pointclouds': pointcloud_batch, # [B, 1024, 3]
        'masks': masks,            # [B, 480, 640]
        'bboxes': bboxes,          # List of [N_i, 8, 3] tensors
        'num_boxes': num_boxes     # [B]
    }