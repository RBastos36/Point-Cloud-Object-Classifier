#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk
import glob
import os
import json
import numpy as np
import webcolors
import open3d as o3d
# from PIL import Image, ImageTk
import cv2

from PointCloud_Learning.dataset_splitter_off import splitDataset
from PointCloud_Learning.main_off import trainModel
from PointCloud_Learning.test_model import testModel
from Part2_Test.All_files_test import getObjects
from PointCloud_Learning.scene_object_identifier import classifyObjects


def getAverageColorName(point_cloud):

    colors = np.asarray(point_cloud.colors)
    color = np.average(colors, axis=0)

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


def getObjectHeight(point_cloud):

    # There is more than one way of getting the BBOX - Choose the one with the best results

    bbox = point_cloud.get_minimal_oriented_bounding_box()
    # bbox = point_cloud.get_axis_aligned_bounding_box()
    # bbox = point_cloud.get_oriented_bounding_box()

    bbox_dim = bbox.get_max_bound() - bbox.get_min_bound()
    height = bbox_dim[1]
    height = round(height*1000)  # milimeters, rounded

    return f"{height} mm"


def getImageFromPoints(point_cloud):

    # Get image_array of the point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(point_cloud)
    render_options = vis.get_render_option()

    render_options.point_size = 50

    vis.poll_events()
    vis.update_renderer()
    image_array = np.asarray(vis.capture_screen_float_buffer())
    vis.destroy_window()


    # Crop & resize image
    height, width, _ = image_array.shape
    min_dim = min(height, width)
    start_row = (height - min_dim) // 2
    start_col = (width - min_dim) // 2
    cropped_image = image_array[start_row:start_row + min_dim, start_col:start_col + min_dim, :]
    resized_image = cv2.resize(cropped_image, (150, 150))
    blur_image = cv2.GaussianBlur(resized_image, (5,5), 0)

    # image_pil = Image.fromarray((resized_image * 255).astype(np.uint8))

    return blur_image


def openResultsWindow(objects):

    # TODO: try replacing opencv window with tkinter top window

    white_row = np.ones((objects[0]['image'].shape[0], 300, 3), dtype=np.uint8) * 255
    image_rows = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, obj in enumerate(objects):
        image = np.hstack([obj['image'], white_row])

        image = cv2.putText(image, obj['label'], (175, 50), font, 1, (0,0,0), 2, cv2.LINE_AA)
        image = cv2.putText(image, 'Color: '+obj['color'], (175, 90), font, 0.7, (0,0,0), 1, cv2.LINE_AA)
        image = cv2.putText(image, 'Height: '+obj['height'], (175, 120), font, 0.7, (0,0,0), 1, cv2.LINE_AA)

        image_rows.append(image)

    image = np.vstack(image_rows)
    cv2.imshow('Results', image)

    while True:
        if cv2.waitKey(50) == 27:   # ESC
            break

        # Close with X button
        elif cv2.getWindowProperty('Results', cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()



# Global variables
root = tk.Tk()


# Main loop
def main():

    def buttonSplitDataset():
        splitDataset()


    def buttonContinueTrain():
        print('Starting train from saved model')
        try:
            trainModel(model_path='models/'+model_name.get(), load_model=True)
        except SystemExit:
            pass


    def buttonNewTrain():
        print('Starting train from zero')
        try:
            trainModel(model_path='models/'+model_name.get(), load_model=False)
        except SystemExit:
            pass


    def buttonTestModel():
        testModel(model_path='models/'+model_name.get(), file_count=100, batch_size=10)


    def buttonOpenScene():
        scene_path = 'data/scenes/pcd/' + selected_scene.get() + '.pcd'
        manual_inputs = bool(checkbox_value.get())

        # Pre-processing: get objects from the scene
        objects = getObjects(scene_path, manual_inputs)

        # Classify objects that were saved as .off
        predicted_labels = classifyObjects(model_path='models/'+model_name.get(), get_metrics=manual_inputs)

        # Update object information
        for i, obj in enumerate(objects):
            if int(obj['idx']) == i:
                obj['label'] = predicted_labels[i]

            obj['color'] = getAverageColorName(obj['points'])
            obj['height'] = getObjectHeight(obj['points'])
            obj['image'] = getImageFromPoints(obj['points'])

        # Display results
        openResultsWindow(objects)


    def buttonOpenCamera():
        print('OPEN CAMERA')    # TODO



    # Get list of scenes
    scenes = glob.glob('data/scenes/pcd/*.pcd')                             # Get paths to all the PCD files in the folder
    scenes = [os.path.splitext(os.path.basename(x))[0] for x in scenes]     # Get the file names (without the extension)
    scenes = sorted(scenes)                                                 # Sort them alphabetically



    # Create base structure: small window with 2 columns (frames)
    root.title("SAVI - Trabalho 2")
    # root.geometry("400x300")
    root.resizable(width=False, height=False)

    frame_train = tk.Frame(root, borderwidth=2, relief="groove", pady=10, padx=15)
    frame_train.grid(row=0, column=0, sticky="nsew")

    frame_scenes = tk.Frame(root, borderwidth=2, relief="groove", pady=10, padx=15)
    frame_scenes.grid(row=0, column=1, sticky="nsew")



    # Column 1: Train Model
    label = tk.Label(frame_train, text="Train Model")
    label.pack()

    model_name = tk.Entry(frame_train, width=18)
    model_name.insert(0, "save.pth")
    model_name.pack(pady=5)

    button = tk.Button(frame_train, text="Split Dataset", command=buttonSplitDataset, width=15)
    button.pack(pady=5)

    button = tk.Button(frame_train, text="Continue Train", command=buttonContinueTrain, width=15)
    button.pack(pady=5)

    button = tk.Button(frame_train, text="New Train", command=buttonNewTrain, width=15)
    button.pack(pady=5)

    button = tk.Button(frame_train, text="Test Model", command=buttonTestModel, width=15)
    button.pack(pady=5)



    # Column 2: Find Objects
    label = tk.Label(frame_scenes, text="Find Objects")
    label.pack()

    selected_scene = tk.StringVar(frame_scenes)
    selected_scene.set(scenes[0])
    dropdown_menu = ttk.Combobox(frame_scenes, textvariable=selected_scene, values=scenes, state="readonly", width=15)
    dropdown_menu.pack(pady=9)

    checkbox_value = tk.IntVar()
    checkbox = tk.Checkbutton(frame_scenes, text="Manual Inputs", variable=checkbox_value)
    checkbox.pack(pady=10)

    button = tk.Button(frame_scenes, text="Open Scene", command=buttonOpenScene, width=15)
    button.pack(pady=5)

    button = tk.Button(frame_scenes, text="Open Camera", command=buttonOpenCamera, width=15)
    button.pack(pady=5)



    # Start the main loop
    tk.mainloop()



if __name__ == '__main__':
    main()