#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk
import glob
import os
import numpy as np
import webcolors
import open3d as o3d
import threading

from gtts import gTTS
from open3d.visualization import gui
from open3d.visualization import rendering

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
    height = bbox_dim[2]
    height = round(height * 1000 * 0.9)  # milimeters, rounded

    return f"{height} mm"


def openResultsWindow(objects):

    # Initialize window
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window("Results", 1024, 768)   # 4x3
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)

    for i, obj in enumerate(objects):

        # Add point cloud
        point_cloud = obj['points']
        scene.scene.add_geometry(f'object_{i}', point_cloud, rendering.MaterialRecord())

        # Add bbox
        bbox = point_cloud.get_minimal_oriented_bounding_box()
        bbox.color = [0, 0, 0]  # colors from obj['color'] are too bright
        scene.scene.add_geometry(f'bbox_{i}', bbox, rendering.MaterialRecord())

        # Add label
        label_pos = bbox.get_center()
        label_pos[2] += 0.15
        label_text = f"{obj['label']}\nColor: {obj['color_name']}\nHeight: {obj['height']}"
        scene.add_3d_label(label_pos, label_text)

    # Run
    scene_bounds = o3d.geometry.AxisAlignedBoundingBox([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    scene.setup_camera(60, scene_bounds, [0, 0, 0])
    gui.Application.instance.run()

def voice(num_objs, str_list ,obj_max, height, color):
    
    file_name = 'Voice_file'

    text = 'In this scene there are ' + str(num_objs) + 'object, ' + str_list + '. The tallest one is the ' + str(obj_max) + ', is ' + str(height) + ' mm tall and its color is ' + color + '.'
    language = 'en'
    tts = gTTS(text=text, lang=language, slow=False)

    # Saving audio file
    tts.save(file_name + '.mp3')
    os.system("play " + file_name + ".mp3"+" tempo 1.2")

    # Deleting audio file
    os.remove(file_name + '.mp3')


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
        heights = []
        obj_labels = []
        colors = []
        for i, obj in enumerate(objects):
            if int(obj['idx']) == i:
                obj['label'] = predicted_labels[i]
                obj_label = obj['label']
                if obj_label == 'cereal':
                    obj_label == 'cereal box'
                elif obj_label == 'coffee':
                    obj_label == ' coffee mug'
                elif obj_label == 'soda':
                    obj_label == 'soda can'

                obj_labels.append(obj_label)

            obj['color_name'] = getAverageColorName(obj['points'])
            color = obj['color_name']
            colors.apped(color)
            obj['height'] = getObjectHeight(obj['points'])
            height = float(obj['height'].split(' ')[0])
            heights.append(height)

        # Calling TTS ------------------
        
        max_height_idx = heights.index(max(heights))
        label = obj_labels[max_height_idx]
        max_height = heights[max_height_idx]
        max_color = colors[max_height_idx]
        num_labels = len(heights)

        dif_objs = list(set(obj_labels))
        counts = []
        for obj in dif_objs:
            count = obj_labels.count(obj)
            counts.append(count)


        string = ""
        for idx, name in enumerate(obj_labels):
            if idx == (len(obj_labels))-1:
                if count[idx] == 1:
                    string = string + "and a " + name
                else:
                    if name == "cereal box":
                        string = string + "and " + str(count) + ' ' + name + "es "
                    else:
                        string = string + "and " + str(count) + ' ' + name + "s "

            elif idx == 0:
                if count[idx] == 1:
                    string = "a " + name
                else:
                    if name == "cereal box":
                        string = str(count[idx]) + " " + name + "es "
                    else:
                        string = str(count[idx]) + " " + name + "s "

            elif 0 < idx < (len(obj_labels)-1):
                if count[idx] == 1:
                    string = string + ", a " + name
                else:
                    if name == "cereal box":
                        string = string + ", " + count[idx] + " " + name + "es "
                    else:
                        string = string + ", " + count[idx] + " " + name + "s "
                    
        
        thread = threading.Thread(target=voice, args=(num_labels, string, label, max_height, max_color))
        thread.start()

        # ------------------------------

        # Display results
        openResultsWindow(objects)


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