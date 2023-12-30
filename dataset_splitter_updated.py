#!/usr/bin/env python3

import glob
import json
from sklearn.model_selection import train_test_split
import random


def splitDataset():

    # Get filenames of all images (including sub-folders)
    image_filenames = glob.glob('datasets/**/*_crop.png', recursive=True)

    train_validation_filenames = glob.glob('datasets/Train_Validation/**/*_crop.png', recursive=True)
    test_filenames = glob.glob('datasets/Test_Only/**/*_crop.png', recursive=True)
    random.shuffle(test_filenames)

    # Check if dataset data exists
    if len(image_filenames) < 1 or len(train_validation_filenames) < 1 or len(test_filenames) < 1:
        raise FileNotFoundError('Dataset files not found')

    # Split datasets - use a rule of 70% train, 20% validation, 10% test
    train_filenames, validation_filenames = train_test_split(train_validation_filenames, test_size=0.3)


    # Print results
    print(f'Total images: {len(image_filenames)}')
    print(f'- {len(train_filenames)} train images')
    print(f'- {len(validation_filenames)} validation images')
    print(f'- {len(test_filenames)} test images')


    # Put results in a dictionary
    output_dict = {
        'train_filenames': train_filenames,
        'validation_filenames': validation_filenames,
        'test_filenames': test_filenames
    }


    # Save dictionary as a JSON file
    json_object = json.dumps(output_dict, indent=2)
    with open('dataset_filenames.json', 'w') as f:
        f.write(json_object)


if __name__ == '__main__':
    splitDataset()
