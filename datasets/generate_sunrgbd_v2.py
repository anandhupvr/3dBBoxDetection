import os
import scipy.io
import numpy as np
from pathlib import Path
from PIL import Image

def parse_sunrgbd_meta(mat_path):
    """Extracts and flattens SUNRGBDMeta3DBB_v2.mat into Python dicts"""
    data = scipy.io.loadmat(mat_path, simplify_cells=True)['SUNRGBDMeta']
    
    processed_scenes = []
    for scene in data:
        # Scene metadata
        scene_data = {
            'scene_path': scene['sequenceName'],
            'Rtilt': scene['Rtilt'],
            'K': scene['K'],
            'depth_path': scene['depthpath'][0].split("sun3d/data/")[-1],
            'rgb_path': scene['rgbpath'][0].split("sun3d/data/")[-1],
            'boxes': []
        }
        

        for i in range(len(scene['groundtruth3DBB'])):
            box_struct = scene['groundtruth3DBB'][i]
            # box = {
            #     'class': box_struct['classname'],
            #     'centroid': box_struct['centroid'],
            #     'size': box_struct['coeffs'] * 2,
            #     'basis': box_struct['basis'],
            #     'orientation': box_struct['orientation']
            # }
            # scene_data.append()
            centroid = box_struct['centroid']
            size = box_struct['coeffs'] * 2
            orientation = box_struct['orientation']
            heading_angle = np.arctan2(-orientation[1], orientation[0])
            label = box_struct['label']
            


    
    return processed_scenes


def depth_to_pointcloud(depth_path, K):
    depth = np.array(Image.open(depth_path)) / 1000.0  # to meters
    u,v = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    uv1 = np.stack([u, v, np.ones_like(u)], axis=-1)
    return (uv1 @ np.linalg.inv(K).T) * depth[..., None]  # Nx3 points


def convert_to_votenet_format(scene_data, output_dir):
    """Converts parsed data to VoteNet-compatible format"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, scene in enumerate(scene_data):
        # 1. Save point cloud (dummy - real PC needs depth processing)
        depth_path = os.path.join('./', scene['depth_path'])
        pc = depth_to_pointcloud(depth_path, scene['K'])
        pc_path = output_dir / f'{i:06d}_pc.npy'
        np.save(pc_path, pc)  # Replace with actual PC
        
        # 2. Save boxes in VoteNet format
        boxes = []
        for box in scene['boxes']:
            boxes.append([
                *box['centroid'],  # cx, cy, cz
                *box['size'],      # l, w, h
                np.arctan2(box['orientation'][1], box['orientation'][0]),  # heading angle
                box['class']       # class name
            ])
        
        np.save(output_dir / f'{i:06d}_bbox.npy', np.array(boxes))
        
        # 3. Save calibration
        with open(output_dir / f'{i:06d}_calib.txt', 'w') as f:
            f.write(f"{' '.join(map(str, scene['Rtilt'].flatten()))}\n")
            f.write(f"{' '.join(map(str, scene['K'].flatten()))}")

meta_data = parse_sunrgbd_meta('SUNRGBDMeta3DBB_v2.mat')
convert_to_votenet_format(meta_data[:10], 'sunrgbd_processed')
