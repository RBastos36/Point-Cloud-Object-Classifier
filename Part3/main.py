#!/usr/bin/env python3

import open3d as o3d
import numpy as np
import webcolors


def getColorName(color):

    # Convert from 0-1 to 0-255
    color = (color * 255).astype(int)

    # Try to get the exact color. If that fails, get the closest color 
    try:
        color_name = webcolors.rgb_to_name(color)

    except ValueError:
        color_name = 'Unknown'
        min_diff = 999999

        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r, g, b = webcolors.hex_to_rgb(key)
            rgb_diff = abs(r - color[0]) + abs(g - color[1]) + abs(b - color[2])

            if rgb_diff < min_diff:
                min_diff = rgb_diff
                color_name = name

    return color_name.capitalize()


def main():

    # Load point cloud
    point_cloud = o3d.io.read_point_cloud('Part3/cluster_2.pcd')
    if len(point_cloud.points) < 1:
        exit('Dataset files not found')

    print(f'Loaded object with {len(point_cloud.points)} points')


    # Bounding box
    bbox = point_cloud.get_minimal_oriented_bounding_box()
    # bbox = point_cloud.get_axis_aligned_bounding_box()
    # bbox = point_cloud.get_oriented_bounding_box()
    bbox_dim = bbox.get_max_bound() - bbox.get_min_bound()
    print('Object dimensions:', bbox_dim)


    # Average color
    colors = np.asarray(point_cloud.colors)
    color = np.mean(colors, axis=0)
    color_name = getColorName(color)
    print(f'Average color: {color} --> {color_name}')


    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()


    # Save image
    # image_array = np.asarray(vis.capture_screen_float_buffer())
    # vis.capture_screen_image("object.png")


    # Destroy window
    vis.destroy_window()
    print('Done!')


    # Test bbox
    bbox.color = (1, 0, 0)
    o3d.visualization.draw_geometries([point_cloud, bbox],
                                    zoom=2.0,
                                    front=[0.7, -0.4, -0.6],
                                    lookat=[0.2, -0.1, 1.2],
                                    up=[-0.5, -0.9, -0.0])


if __name__ == '__main__':
    main()

