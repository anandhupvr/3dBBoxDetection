import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from sklearn.model_selection import train_test_split

class SunRGBDDataset(Dataset):
    def __init__(self, root_dir, split='train', val_ratio=0.1, test_ratio=0.1, random_seed=42):
        """
        Args:
            root_dir (str): Directory containing .npz files
            split (str): 'train', 'val', or 'test'
            val_ratio (float): Fraction of data for validation
            test_ratio (float): Fraction of data for testing
            random_seed (int): For reproducible splits
        """
        self.root_dir = root_dir
        self.split = split
        
        # Get all scene files and sort them
        all_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.npz')])
        
        # Create splits
        train_files, test_files = train_test_split(
            all_files, 
            test_size=test_ratio, 
            random_state=random_seed
        )
        train_files, val_files = train_test_split(
            train_files, 
            test_size=val_ratio/(1-test_ratio), 
            random_state=random_seed
        )
        
        # Assign files based on split
        if split == 'train':
            self.scene_files = train_files
        elif split == 'val':
            self.scene_files = val_files
        else:  # test
            self.scene_files = test_files
            
        print(f"{split} set: {len(self.scene_files)} scenes")

    def __len__(self):
        return len(self.scene_files)\
        
    def __getitem__(self, idx):
        data = np.load(os.path.join(self.root_dir, self.scene_files[idx]), allow_pickle=True)
        
        import pdb; pdb.set_trace()
        rgb_path = data['paths'][()]['rgb']
        depth_path = data['paths'][()]['depth']
        seg_path = data['paths'][()]['seg']

        points3d = data['points3d']
        valid_mask = ~np.isnan(points3d).any(axis=-1)
        points3d = points3d[valid_mask]

        # for box in data['boxes']:


        # print(data)
        return rgb_path
