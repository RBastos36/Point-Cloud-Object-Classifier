#!/usr/bin/env python3


import json
from dataset import Dataset
import torch
import matplotlib.pyplot as plt

from model import Model
from trainer import Trainer


def main():

    # -----------------------------------------------------------------
    # Hyperparameters initialization
    # -----------------------------------------------------------------
    batch_size = 100
    learning_rate = 0.001
    num_epochs = 50


    # -----------------------------------------------------------------
    # Prepare Datasets
    # -----------------------------------------------------------------

    # Reading from json file
    with open('dataset_filenames.json', 'r') as f:
        dataset_filenames = json.load(f)

    train_filenames = dataset_filenames['train_filenames']
    validation_filenames = dataset_filenames['validation_filenames']

    train_filenames = train_filenames[0:500]
    validation_filenames = validation_filenames[0:200]

    # While testing, use smaller datasets
    train_filenames = train_filenames[0:1000]
    validation_filenames = validation_filenames[0:200]

    print(f'Used {len(train_filenames)} for training and {len(validation_filenames)} for validation.')

    train_dataset = Dataset(train_filenames)
    validation_dataset = Dataset(validation_filenames)

    # Try the train_dataset
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)


    # -----------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------
    model = Model()
    trainer = Trainer(model=model,
                      train_loader=train_loader,
                      validation_loader=validation_loader,
                      learning_rate=learning_rate,
                      num_epochs=num_epochs,
                      model_path='models/checkpoint.pkl',
                      load_model=False)
    trainer.train()

    plt.show()


if __name__ == "__main__":
    main()
