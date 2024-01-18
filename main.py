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
from PreProcessing.All_files_processing import getObjects
from PointCloud_Learning.scene_object_identifier import classifyObjects
from Camera.camera import initCamera


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


def getObjectDimensions(point_cloud):

    # There is more than one way of getting the BBOX - Choose the one with the best results

    bbox = point_cloud.get_minimal_oriented_bounding_box()
    # bbox = point_cloud.get_axis_aligned_bounding_box()
    # bbox = point_cloud.get_oriented_bounding_box()

    bbox_dim = bbox.get_max_bound() - bbox.get_min_bound()
    height = bbox_dim[2]
    height = round(height * 1000 * 0.9)  # milimeters, rounded

    width = max(bbox_dim[0], bbox_dim[1])
    width = round(width * 1000 * 0.9)

    return height, width


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
        label_text = f"{obj['label'].capitalize()}\nColor: {obj['color_name']}\nHeight: {obj['height']} mm\nWidth: {obj['width']} mm"
        scene.add_3d_label(label_pos, label_text)

    # Run
    scene_bounds = o3d.geometry.AxisAlignedBoundingBox([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    scene.setup_camera(60, scene_bounds, [0, 0, 0])
    gui.Application.instance.run()


def voice(num_objs, str_list ,obj_max, height, color, closest_to_center):
    
    file_name = 'Voice_file'

    if num_objs > 1:
        text = 'In this scene there are ' + str(num_objs) + ' objects: '
    else:
        text = 'In this scene there is one object: '
    text += str_list + \
        '. The tallest one is the ' + str(obj_max) + ', it is ' + str(round(height)) + ' mm tall and its color is ' + color + \
        '. The object closest to the middle of the table is a ' + closest_to_center + '.' 
    language = 'en'
    print('\nTTS: ' + text + '\n')
    tts = gTTS(text=text, lang=language, slow=False)

    # Saving audio file
    tts.save(file_name + '.mp3')
    os.system("play " + file_name + ".mp3"+" tempo 1.2")

    # Deleting audio file
    file_name += '.mp3'
    if os.path.exists(file_name):
        os.remove(file_name)


def analyseScene(scene_path, model_path, manual_inputs, camera_image=False):

    # Pre-processing: get objects from the scene
    if not camera_image:
        objects = getObjects(scene_path, cluster_eps=0.031, cluster_minpoints=74, ask_for_input=manual_inputs)
    else:
        objects = getObjects(scene_path, cluster_eps=0.1, cluster_minpoints=40, ask_for_input=False)

    print(f'{len(objects)} objects detected!')

    if len(objects) < 1:
        return

    # Classify objects that were saved as .off
    predicted_labels = classifyObjects(model_path=model_path, get_metrics=manual_inputs)

    # Update object information
    heights = []
    obj_labels = []
    colors = []
    for i, obj in enumerate(objects):
        if int(obj['idx']) == i:
            obj['label'] = predicted_labels[i]
            if obj['label'] == 'cereal':
                obj['label'] = 'cereal box'
            elif obj['label'] == 'coffee':
                obj['label'] = 'coffee mug'
            elif obj['label'] == 'soda':
                obj['label'] = 'soda can'

            obj_labels.append(obj['label'])

        obj['color_name'] = getAverageColorName(obj['points'])
        color = obj['color_name']
        colors.append(color)
        obj['height'], obj['width'] = getObjectDimensions(obj['points'])
        obj['dist_to_center'] = np.linalg.norm(obj['center'] - np.array([0, 0, 0]))

        heights.append(obj['height'])

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
    for idx, name in enumerate(dif_objs):
        if idx == (len(dif_objs) - 1):
            if counts[idx] == 1:
                string = string + " and a " + name
            else:
                if name == "cereal box":
                    string = string + " and " + str(counts[idx]) + ' ' + name + "es"
                else:
                    string = string + " and " + str(counts[idx]) + ' ' + name + "s"

        elif idx == 0:
            if counts[idx] == 1:
                string = "a " + name
            else:
                if name == "cereal box":
                    string = str(counts[idx]) + " " + name + "es"
                    string = str(counts[idx]) + " " + name + "es"
                else:
                    string = str(counts[idx]) + " " + name + "s"

        elif 0 < idx < (len(dif_objs) - 1):
            if counts[idx] == 1:
                string = string + ", a " + name
            else:
                if name == "cereal box":
                    string = string + ", " + str(counts[idx]) + " " + name + "es"
                else:
                    string = string + ", " + str(counts[idx]) + " " + name + "s"

    if len(dif_objs) == 1:
        name = dif_objs[0]
        if counts[0] > 1:
            if name == "cereal box":
                name += "es"
            else:
                name += "s"

        if counts[0] == 1:
            string = "a " + name
        else:
            string = str(counts[0]) + " " + name

    # Find closest to center
    min_entry = min(objects, key=lambda x: x["dist_to_center"])
    closest_object = min_entry["label"]

    thread = threading.Thread(target=voice, args=(num_labels, string, label, max_height, max_color, closest_object))
    thread.start()

    # ------------------------------

    # Display results
    openResultsWindow(objects)


def validateIntegerInput(new_value):
    if new_value == "":
        return True  # Allow empty input
    try:
        return int(new_value) > 0
    except ValueError:
        return False



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
        try:
            batch_size = int(input_batch_size.get())
            file_count = int(input_file_count.get())
        except ValueError:
            print('Invalid input. Batch size and file count must be integers')

        testModel(model_path='models/'+model_name.get(), file_count=file_count, batch_size=batch_size, metrics_averaging=str(metrics_variable.get()))


    def buttonOpenScene():
        scene_path = 'data/scenes/pcd/' + selected_scene.get() + '.pcd'
        manual_inputs = bool(checkbox_value.get())
        model_path = 'models/'+model_name.get()
        analyseScene(scene_path, model_path, manual_inputs)


    def buttonViewScene():
        scene_path = 'data/scenes/pcd/' + selected_scene.get() + '.pcd'
        pcd = o3d.io.read_point_cloud(scene_path)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=np.array([0., 0., 0.]))

        view = {
            "class_name" : "ViewTrajectory",
            "interval" : 29,
            "is_loop" : False,
            "trajectory" : 
            [
                {
                    "boundingbox_max" : [ 2.7116048336029053, 1.2182252407073975, 3.8905272483825684 ],
                    "boundingbox_min" : [ -2.4257750511169434, -1.6397310495376587, -1.3339539766311646 ],
                    "field_of_view" : 60.0,
                    "front" : [ 0.1679443468780758, -0.33027614037477337, -0.92882310880535257 ],
                    "lookat" : [ 0.14291489124298096, -0.21075290441513062, 1.2782866358757019 ],
                    "up" : [ -0.15419696802726832, -0.93940923109371266, 0.30615942184933193 ],
                    "zoom" : 0.59999999999999987
                }
            ],
            "version_major" : 1,
            "version_minor" : 0
        }

        gui.Application.instance.initialize()
        window = gui.Application.instance.create_window("Open3D", 1024, 768)   # 4x3
        scene = gui.SceneWidget()
        scene.scene = rendering.Open3DScene(window.renderer)
        window.add_child(scene)

        scene.scene.add_geometry(f'pcd', pcd, rendering.MaterialRecord())
        scene.scene.add_geometry(f'origin', origin, rendering.MaterialRecord())

        scene.setup_camera(60, pcd.get_axis_aligned_bounding_box(), [0, 0, 0])

        scene.look_at(np.array(view['trajectory'][0]['lookat'], dtype=np.float32),
                    np.array(view['trajectory'][0]['front'], dtype=np.float32),
                    np.array(view['trajectory'][0]['up'], dtype=np.float32))
        gui.Application.instance.run()

        # o3d.visualization.draw_geometries([pcd, origin],
        #                                 zoom=view['trajectory'][0]['zoom'],
        #                                 front=view['trajectory'][0]['front'],
        #                                 lookat=view['trajectory'][0]['lookat'],
        #                                 up=view['trajectory'][0]['up'])


    def buttonOpenCamera():
        pcd_path = 'Camera/temp.pcd'
        try:
            initCamera(pcd_path)
        except SystemExit:
            return
        model_path = 'models/'+model_name.get()
        analyseScene(pcd_path, model_path, manual_inputs=False, camera_image=True)


    # Get list of scenes
    scenes = glob.glob('data/scenes/pcd/*.pcd')                             # Get paths to all the PCD files in the folder
    scenes = [os.path.splitext(os.path.basename(x))[0] for x in scenes]     # Get the file names (without the extension)
    scenes = sorted(scenes)                                                 # Sort them alphabetically



    # Create base structure: small window with 2 columns (frames)
    root = tk.Tk()
    root.title("SAVI - Trabalho 2")
    # root.geometry("400x300")
    root.resizable(width=False, height=False)

    frame_model = tk.Frame(root, borderwidth=2, relief="groove", pady=10)
    frame_model.grid(row=0, column=0, columnspan=3, sticky="nsew")

    frame_train = tk.Frame(root, borderwidth=2, relief="groove", pady=10, padx=15)
    frame_train.grid(row=1, column=0, sticky="nsew")

    frame_test = tk.Frame(root, borderwidth=2, relief="groove", pady=10, padx=15)
    frame_test.grid(row=1, column=1, sticky="nsew")

    frame_scenes = tk.Frame(root, borderwidth=2, relief="groove", pady=10, padx=15)
    frame_scenes.grid(row=1, column=2, sticky="nsew")


    # Top bar: model path
    tk.Label(frame_model, text="").grid(row=0, column=0, padx=70)

    label = tk.Label(frame_model, text="Selected Model:")
    label.grid(row=0, column=1, pady=5, padx=5)

    model_name = tk.Entry(frame_model, width=15)
    model_name.insert(0, "save_8.pth")
    model_name.grid(row=0, column=2)


    # Column 1: Train Model
    label = tk.Label(frame_train, text="Train Model")
    label.pack()

    button = tk.Button(frame_train, text="Split Dataset", command=buttonSplitDataset, width=15)
    button.pack(pady=5)

    button = tk.Button(frame_train, text="Continue Train", command=buttonContinueTrain, width=15)
    button.pack(pady=5)

    button = tk.Button(frame_train, text="New Train", command=buttonNewTrain, width=15)
    button.pack(pady=5)


    # Column 2: Test Model
    label = tk.Label(frame_test, text="Test Model")
    label.grid(row=0, column=0, columnspan=2)

    label = tk.Label(frame_test, text="Batch size:")
    label.grid(row=1, column=0, pady=8)

    input_batch_size = tk.Entry(frame_test, validate="key", width=5, validatecommand=(frame_test.register(validateIntegerInput), '%P'))
    input_batch_size.insert(0, "10")
    input_batch_size.grid(row=1, column=1, pady=8)

    label = tk.Label(frame_test, text="File count:")
    label.grid(row=2, column=0, pady=5)

    input_file_count = tk.Entry(frame_test, validate="key", width=5, validatecommand=(frame_test.register(validateIntegerInput), '%P'))
    input_file_count.insert(0, "100")
    input_file_count.grid(row=2, column=1, pady=5)


    # Radio button to choose between 'macro' and 'micro' averaging in precision and recall metrics
    tk.Label(frame_test, text="Metrics Averaging:").grid(row=3, column=0, columnspan=2, pady=5)

    metrics_variable = tk.StringVar(frame_test, "macro")
    radiobtn = ttk.Radiobutton(frame_test, text = "macro", variable = metrics_variable, value = "macro")
    radiobtn.grid(row=4, column=0, columnspan=2, pady=5)
    radiobtn = ttk.Radiobutton(frame_test, text = "micro", variable = metrics_variable, value = "micro")
    radiobtn.grid(row=5, column=0, columnspan=2, pady=5)


    button = tk.Button(frame_test, text="Start Test", command=buttonTestModel, width=15)
    button.grid(row=6, column=0, columnspan=2, pady=5)


    # Column 3: Find Objects
    label = tk.Label(frame_scenes, text="Find Objects in Scene")
    label.pack()

    selected_scene = tk.StringVar(frame_scenes)
    selected_scene.set(scenes[0])
    dropdown_menu = ttk.Combobox(frame_scenes, textvariable=selected_scene, values=scenes, state="readonly", width=15)
    dropdown_menu.pack(pady=9)

    button = tk.Button(frame_scenes, text="View Full Scene", command=buttonViewScene, width=15)
    button.pack(pady=5)

    checkbox_value = tk.IntVar()
    checkbox = tk.Checkbutton(frame_scenes, text="Manual Inputs", variable=checkbox_value)
    checkbox.pack(pady=10)

    button = tk.Button(frame_scenes, text="Open Scene", command=buttonOpenScene, width=15)
    button.pack(pady=5)

    button = tk.Button(frame_scenes, text="Open Camera", command=buttonOpenCamera, width=15)
    button.pack(pady=5,side='bottom')


    # Start the main loop
    tk.mainloop()



if __name__ == '__main__':
    main()