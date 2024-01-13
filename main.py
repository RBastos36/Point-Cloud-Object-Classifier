#!/usr/bin/env python3

import tkinter as tk
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


    window = tk.Tk()
    window.geometry("300x400")
    window.resizable(width=False, height=False)
    window.title("SAVI - Trabalho 2")

    frame_train = tk.Frame(window)
    frame_train.pack(side=tk.TOP, pady=10)

    label = tk.Label(frame_train, text="Train Model", anchor="w", justify="left")
    label.pack()

    button = tk.Button(frame_train, text="Split Dataset", command=buttonSplitDataset, width=15)
    button.pack(pady=5)

    button = tk.Button(frame_train, text="Continue Train", command=buttonContinueTrain, width=15)
    button.pack(pady=5)

    button = tk.Button(frame_train, text="New Train", command=buttonNewTrain, width=15)
    button.pack(pady=5)

    button = tk.Button(frame_train, text="Test Model", command=buttonTestModel, width=15)
    button.pack(pady=5)


    # Start the main loop
    window.mainloop()



if __name__ == '__main__':
    main()