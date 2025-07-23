import numpy as np
import scipy.io
from PIL import Image
import os
from multiprocessing import Pool
from tqdm import tqdm
import sys


def validate_scene_data(points3d_rgb, boxes_np, seg_mask):
    """Validate all data before saving"""
    checks = {
        'Point cloud has NaN values': not np.isnan(points3d_rgb).any(),
        'Point cloud shape invalid': points3d_rgb.shape[1] == 6,
        'No boxes in scene': len(boxes_np) > 0,
        'Boxes have NaN values': not np.isnan(boxes_np).any(),
        'Segmentation mask empty': seg_mask.size > 0,
    }
    
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        return False, failed
    return True, []



def process_scene(imageId):
    try:
        # Load metadata
        # meta = scipy.io.loadmat('SUNRGBDMeta3DBB_v2.mat')['SUNRGBDMeta'][0, 0]
        meta = meta_data[imageId-1]
        
        # Load and process depth
        depth_path = os.path.join('./', meta['depthpath'][0].split("sun3d/data/")[-1])

        # Load depth and create point cloud
        depth = np.array(Image.open(depth_path)).astype(np.float32) / 1000.0
        K = meta['K'].reshape(3,3)
        v, u = np.indices(depth.shape)
        points3d = (np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(K).T) * depth[..., None]
        
        # Attach RGB
        rgb_path = os.path.join('./', meta['rgbpath'][0].split("sun3d/data/")[-1])
        rgb = np.array(Image.open(rgb_path)) / 255.0
        points3d_rgb = np.concatenate([points3d.reshape(-1, 3), rgb.reshape(-1, 3)], axis=1)
        points3d_rgb = points3d_rgb[~np.isnan(points3d_rgb[:, 0])]
        seg_path = os.path.join('./', meta[0][0], 'seg.mat')
        seg_data = scipy.io.loadmat(seg_path)
        seg_mask = seg_data['seglabel'].astype(np.uint8)
        seg_names = seg_data['names'][0] 
        
        # Create mapping from label IDs to names
        label_map = {}
        for i, name_array in enumerate(seg_names):
            label_map[i+1] = str(name_array[0])  # Labels start at 1
        
        # for now saving only seg_mask
        seg_output = {
            'mask': seg_mask, 
            'label_names': label_map, 
            'original_path': seg_path
        }

        boxes = []
        for j in range(len(meta['groundtruth3DBB'][0])):
            box = meta['groundtruth3DBB'][0,j]

            # orientation = box['orientation'][0] if box['orientation'].size == 3 else box['orientation'][0][0]
            orientation = box['orientation'][0] 
            
            # centroid = box['centroid'][0] if box['centroid'].size == 3 else box['centroid'][0][0]
            centroid = box['centroid'][0]

            # coeffs = box['coeffs'][0] if box['coeffs'].size == 3 else box['coeffs'][0][0]
            coeffs = box['coeffs'][0]

            # label = box['label'][0][0] if isinstance(box['label'][0], np.ndarray) else box['label'][0]
            label = box['label'][0][0]

            max_val = 1e300  # Practical upper limit
            orientation = np.clip(orientation, -max_val, max_val)


            boxes.append([
                float(centroid[0]), float(centroid[1]), float(centroid[2]),  # x,y,z
                float(coeffs[1]*2), float(coeffs[0]*2), float(coeffs[2]*2),  # l,w,h
                float(-np.arctan2(orientation[1], orientation[0])),  # rotation
                float(label)  # class ID
            ])
 
        boxes_np = np.array(boxes, dtype=np.float32)

        is_valid, validation_errors = validate_scene_data(points3d_rgb, boxes_np, seg_mask)
        if not is_valid:
            print(f"\nInvalid scene {imageId}: {', '.join(validation_errors)}")
            return False

        # Save
        output_id = f"{imageId:06d}"


        output_id = f"{imageId:06d}"
        save_data = {
            'points3d': points3d.astype(np.float32),
            'boxes': boxes_np,
            'calib': {
                'K': K,
                'Rtilt': meta['Rtilt']
            },
            'paths': {
                'rgb': rgb_path,
                'depth': depth_path,
                'seg': seg_path
            }
        }
        
        np.savez_compressed(f'sunrgbd_trainval/{output_id}.npz', **save_data)

        # np.save(f'sunrgbd_trainval/depth/{output_id}.npy', points3d_rgb)
        # np.save(f'sunrgbd_trainval/label/{output_id}_bbox.npy', boxes_np)
        # shutil.copy(rgb_path, f'sunrgbd_trainval/image/{output_id}.jpg')
        # np.savetxt(f'sunrgbd_trainval/calib/{output_id}.txt', 
        #           np.concatenate([meta['Rtilt'].flatten(), K.flatten()]))
        
        # saving only the mask
        output_id = f"{imageId:06d}"
        # np.save(f'sunrgd_trainval/seg/{output_id}_seg.npz', seg_output['mask'])
        # np.savez_compressed(f'sunrgbd_trainval/seg/{output_id}_seg.npz', mask=seg_output['mask'])
        
        return True

    except Exception as e:
        print(f"Error processing imageId {imageId}: {str(e)}")
        return False


def init_worker():
    global meta_data
    meta_data = scipy.io.loadmat('SUNRGBDMeta3DBB_v2.mat')['SUNRGBDMeta'][0]

def main():
    # Create output directories
    # os.makedirs('sunrgbd_trainval/depth', exist_ok=True)
    # os.makedirs('sunrgbd_trainval/image', exist_ok=True)
    # os.makedirs('sunrgbd_trainval/calib', exist_ok=True)
    # os.makedirs('sunrgbd_trainval/label', exist_ok=True)
    # os.makedirs('sunrgbd_trainval/seg', exist_ok=True)

    os.makedirs('sunrgbd_trainval', exist_ok=True)

    total_scenes = 10335
    success_count = 0

    with tqdm(total=total_scenes, desc="Processing Scenes", file=sys.stdout) as pbar:
        with Pool(processes=8, initializer=init_worker) as pool:
            for result in pool.imap(process_scene, range(1, total_scenes)):
                success_count += int(result)
                pbar.update()
                pbar.set_postfix({'Success': success_count, 'Failed': pbar.n - success_count})

        print(f"\nProcessing complete. Success: {success_count}/{total_scenes}")

if __name__ == '__main__':
    main()