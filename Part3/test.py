#!/usr/bin/env python3

import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageTk


def getImage(point_cloud):

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(point_cloud)
    render_options = vis.get_render_option()

    render_options.point_size = 50

    vis.poll_events()
    vis.update_renderer()
    image_array = np.asarray(vis.capture_screen_float_buffer())
    vis.destroy_window()

    return image_array




def main():

    # Load point cloud
    point_cloud = o3d.io.read_point_cloud('Part3/cluster_2.pcd')
    if len(point_cloud.points) < 1:
        exit('File not found')
    print(f'Loaded object with {len(point_cloud.points)} points')


    # Show with open3d
    o3d.visualization.draw_geometries([point_cloud])


    # Get image from point cloud
    image = getImage(point_cloud)
    print(f'\nAverage color: {np.mean(image, axis=(0, 1))}\n')


    # Show image with opencv
    cv2.imshow('Results', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # Show image with matplotlib
    plt.imshow(image)
    plt.show()


    # Show image with tkinter
    image = (image * 255).astype(np.uint8)
    root = tk.Tk()
    root.title("Image Display")
    pil_img = Image.fromarray(image)
    tk_img = ImageTk.PhotoImage(pil_img)
    label = tk.Label(root, image=tk_img)
    label.pack()
    root.mainloop()





if __name__ == '__main__':
    main()