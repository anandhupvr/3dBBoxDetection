import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import scipy.io

from sunrgbd_utils import my_compute_box_3d

def visualize(sunrgbd_data):

    data = np.load(os.path.join('./sunrgbd_trainval', sunrgbd_data), allow_pickle=True)

    rgb_path = data['paths'][()]['rgb']
    depth_path = data['paths'][()]['depth']
    seg_path = data['paths'][()]['seg']

    rgb = plt.imread(rgb_path)
    seg_data = scipy.io.loadmat(seg_path)
    seg_mask = seg_data['seglabel'].astype(np.uint8)
    seg_names = seg_data['names'][0]

    points3d = data['points3d']
    valid_mask = ~np.isnan(points3d).any(axis=-1)
    points3d = points3d[valid_mask]

    # import pdb; pdb.set_trace()
    pl = pv.Plotter()
    pl.add_points(points3d, color="red", point_size=2, opacity=0.5)
    for box in data['boxes']:
        center = box[:3]
        size = box[3:6] * 2
        angle = box[6]
        corners = my_compute_box_3d(center, size, angle)
        # Define box edges (12 lines connecting the 8 corners)
        edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ])
        
        # Create a PyVista mesh for the box
        box_mesh = pv.PolyData()
        box_mesh.points = corners
        box_mesh.lines = np.hstack([2 * np.ones((12, 1)), edges]).astype(int)  # Format: [num_pts, pt1, pt2, ...]
        
        # Add the box to the plotter
        pl.add_mesh(box_mesh, color="blue", line_width=3, opacity=0.8)


    plt.imshow(rgb)
    print(seg_names)
    plt.imshow(seg_mask)
    plt.show()

    # import pdb; pdb.set_trace()

    # plot 
    pl.show()



def main():
    data_dir = "./sunrgbd_trainval"
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
    data_list = np.random.choice(all_files)
    visualize(data_list)

if __name__ == "__main__":
    main()
