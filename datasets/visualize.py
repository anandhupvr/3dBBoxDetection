import numpy as np
import scipy.io
from PIL import Image
import os
import pyvista as pv
from itertools import product


def load_sunrgbd_depth(depth_path):
    depth_vis = np.array(Image.open(depth_path))  # 16-bit PNG
    
    # SUN RGB-D specific decoding:
    depth_inpaint = np.bitwise_or(
        np.right_shift(depth_vis, 3),
        np.left_shift(depth_vis, 16-3))
    
    depth_inpaint = depth_inpaint.astype(np.float32)/1000.0  # Convert to meters
    depth_inpaint[depth_inpaint > 8.0] = 8.0  # Clip to 8m (Kinect's max reliable range)
    return depth_inpaint


def backproject_depth(depth, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    # SUN RGB-D coordinate convention: X right, Y down, Z forward
    points = np.stack((X, Z, -Y), axis=-1)  # Note the -Y and swapping Y/Z
    return points.reshape(-1, 3)  # Flatten to (N, 3)

def get_box_corners(centroid, basis, coeffs):
    signs = np.array(list(product([-1, 1], repeat=3)))  # 8 combinations
    corners = centroid + (basis @ (signs * coeffs).T).T
    return corners

def process_scene(imageId):
    # Load metadata
    meta_data = scipy.io.loadmat('SUNRGBDMeta3DBB_v2.mat')['SUNRGBDMeta'][0]
    meta = meta_data[imageId-1]
    
    # Load and process depth
    depth_path = os.path.join('./', meta['depthpath'][0].split("sun3d/data/")[-1])
    # depth = np.array(Image.open(depth_path)).astype(np.float32) / 1000.0
    depth = load_sunrgbd_depth(depth_path)
    
    K = meta['K'].reshape(3, 3)
    Rtilt = meta['Rtilt'].reshape(3, 3) 

    print(f"Depth image shape: {depth.shape}")
    print(f"Depth range: {depth[depth > 0].min():.3f} - {depth[depth > 0].max():.3f} meters")

    points_cam = backproject_depth(depth, K)
    points_world = (Rtilt @ points_cam.T).T  # Apply Rtilt rotation
    
    # Create plot
    plotter = pv.Plotter()
    point_cloud = pv.PolyData(points_world)
    plotter.add_mesh(point_cloud, color='red', point_size=2, 
                    render_points_as_spheres=False, opacity=0.8)
    
    # Add bounding boxes
    for j in range(len(meta['groundtruth3DBB'][0])):
        box_data = meta['groundtruth3DBB'][0, j]
        centroid = np.array(box_data['centroid'][0]).reshape(3)
        basis = np.array(box_data['basis']).reshape(3, 3)
        coeffs = np.array(box_data['coeffs']).reshape(3)
        
        corners = get_box_corners(centroid, basis, coeffs)
        
        # Define edges for box visualization
        edges = [
            (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),
            (4,5),(4,6),(5,7),(6,7)
        ]
        
        for e0, e1 in edges:
            pts = np.array([corners[e0], corners[e1]])
            plotter.add_lines(pts, color='green', width=4)
    
    plotter.show()

def main():
    imageId = 1
    process_scene(imageId)

if __name__ == '__main__':
    main()