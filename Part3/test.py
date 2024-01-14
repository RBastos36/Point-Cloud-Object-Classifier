#!/usr/bin/env python3

import open3d as o3d
import numpy as np
import cv2


def main():

    # Load point cloud
    point_cloud = o3d.io.read_point_cloud('Part3/cluster_2.pcd')
    if len(point_cloud.points) < 1:
        exit('Dataset files not found')

    print(f'Loaded object with {len(point_cloud.points)} points')


    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(point_cloud)
    render_options = vis.get_render_option()

    render_options.point_size = 50

    vis.poll_events()
    vis.update_renderer()
    image_array = np.asarray(vis.capture_screen_float_buffer())
    vis.destroy_window()

    cv2.imshow('Results', image_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()

