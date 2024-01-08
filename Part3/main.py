#!/usr/bin/env python3

import open3d as o3d
import numpy as np


# TODO: try using cv2.projectPoints


def main():

    # Load point cloud
    point_cloud = o3d.io.read_point_cloud('cluster_2.pcd')

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()

    # Save image
    # image_array = np.asarray(vis.capture_screen_float_buffer())
    vis.capture_screen_image("object.png")

    # Destroy window
    vis.destroy_window()
    print('Done!')



if __name__ == '__main__':
    main()

