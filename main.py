#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk
import glob
import os
import json
import numpy as np
import webcolors

from PointCloud_Learning.dataset_splitter_off import splitDataset
from PointCloud_Learning.main_off import trainModel
from PointCloud_Learning.test_model import testModel
from Part2_Test.All_files_test import getObjects
from PointCloud_Learning.scene_object_identifier import classifyObjects


def getAverageColorName(point_cloud):

    colors = np.asarray(point_cloud.colors)
    color = np.mean(colors, axis=0)

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


def getObjectDimentions(point_cloud):

    # There is more than one way of getting the BBOX - Choose the one with the best results

    bbox = point_cloud.get_minimal_oriented_bounding_box()
    # bbox = point_cloud.get_axis_aligned_bounding_box()
    # bbox = point_cloud.get_oriented_bounding_box()

    bbox_dim = bbox.get_max_bound() - bbox.get_min_bound()
    height = bbox_dim[1]
    height = round(height*1000)  # milimeters, rounded

    return f"{height} mm"



def main():

    def buttonSplitDataset():
        splitDataset()


    def buttonContinueTrain():
        print('Starting train from saved model')
        try:
            trainModel(model_path='models/save.pth', load_model=True)
        except SystemExit:
            pass


    def buttonNewTrain():
        print('Starting train from zero')
        try:
            trainModel(model_path='models/save.pth', load_model=False)
        except SystemExit:
            pass


    def buttonTestModel():
        testModel(model_path='models/save.pth', file_count=100, batch_size=10)


    def buttonOpenScene():
        scene_path = 'data/scenes/pcd/' + selected_scene.get() + '.pcd'
        manual_inputs = bool(checkbox_value.get())

        # Pre-processing: get objects from the scene
        objects = getObjects(scene_path, manual_inputs)

        # Classify objects that were saved as .off
        predicted_labels = classifyObjects(manual_inputs)

        # Update object information
        for i, obj in enumerate(objects):
            if int(obj['idx']) == i:
                obj['label'] = predicted_labels[i]
            obj['color'] = getAverageColorName(obj['points'])
            obj['height'] = getObjectDimentions(obj['points'])
            

            # Temporary - these don't work with JSON
            del obj['points']
            del obj['center']

        # Display results
        print(json.dumps(objects, indent=4))

        # TODO: open a new window with an image of each point cloud + information


    def buttonOpenCamera():
        print('OPEN CAMERA')    # TODO



    # Get list of scenes
    scenes = glob.glob('data/scenes/pcd/*.pcd')                             # Get paths to all the PCD files in the folder
    scenes = [os.path.splitext(os.path.basename(x))[0] for x in scenes]     # Get the file names (without the extension)
    scenes = sorted(scenes)                                                 # Sort them alphabetically



    # Create base structure: small window with 2 columns (frames)
    root = tk.Tk()
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
    root.mainloop()



if __name__ == '__main__':
    main()