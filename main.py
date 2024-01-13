#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk
import glob
import os

from PointCloud_Learning.dataset_splitter_off import splitDataset
from PointCloud_Learning.main_off import trainModel
from PointCloud_Learning.test_model import testModel


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
        file_path = 'data/scenes/pcd/' + selected_scene.get() + '.pcd'

        # TODO: run the scene thing with this path
        print(file_path)

    def buttonOpenCamera():
        print('OPEN CAMERA')



    # Get list of scenes
    scenes = glob.glob('data/scenes/pcd/*.pcd')                             # Get paths to all the PCD files in the folder
    scenes = [os.path.splitext(os.path.basename(x))[0] for x in scenes]     # Get the file names (without the extension)
    scenes = sorted(scenes)                                                 # Sort them alphabetically



    # Create base structure: small window with 2 columns (frames)
    root = tk.Tk()
    root.title("SAVI - Trabalho 2")
    # root.geometry("400x300")
    root.resizable(width=False, height=False)

    frame_train = tk.Frame(root, borderwidth=2, relief="groove", pady=10, padx=10)
    frame_train.grid(row=0, column=0, sticky="nsew")

    frame_scenes = tk.Frame(root, borderwidth=2, relief="groove", pady=10, padx=10)
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
    dropdown_menu.pack(pady=10)

    button = tk.Button(frame_scenes, text="Open Scene", command=buttonOpenScene, width=15)
    button.pack(pady=5)

    button = tk.Button(frame_scenes, text="Open Camera", command=buttonOpenCamera, width=15)
    button.pack(pady=5)



    # Start the main loop
    root.mainloop()



if __name__ == '__main__':
    main()