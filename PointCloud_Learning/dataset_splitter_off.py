#!/usr/bin/env python3

import glob
import json
from sklearn.model_selection import train_test_split
import random


def splitDataset():

    # Get filenames of all images (including sub-folders)
    pts_filenames = glob.glob('data/objects_off/*.off', recursive=True)


    # Check if dataset data exists
    if len(pts_filenames) < 1:
        raise FileNotFoundError('Dataset files not found')


    # Previously "Train_Validation" folder
    train_validation_objects = [
        'bowl_1',
        'bowl_2',
        'bowl_3',
        'bowl_4',
        'bowl_5',
        'cap_1',
        'cap_2',
        'cap_3',
        'cereal_box_1',
        'cereal_box_2',
        'cereal_box_3',
        'cereal_box_4',
        'coffee_mug_1',
        'coffee_mug_2',
        'coffee_mug_3',
        'coffee_mug_4',
        'coffee_mug_5',
        'coffee_mug_6',
        'soda_can_1',
        'soda_can_2',
        'soda_can_3',
        'soda_can_4',
        'soda_can_5'
    ]


    # Previously "Test_Only" folder
    test_objects = [
        'bowl_6',
        'cap_4',
        'cereal_box_5',
        'coffee_mug_7',
        'coffee_mug_8',
        'soda_can_6'
    ]


    # Create smaller lists from image_filenames, with the objects from train_validation_objects and test_objects
    train_validation_filenames = [filename for filename in pts_filenames for obj in train_validation_objects if f'/{obj}' in filename]
    test_filenames = [filename for filename in pts_filenames for obj in test_objects if f'/{obj}' in filename]
    

    # Split & shuffle test, train and validation datasets
    train_filenames, validation_filenames = train_test_split(train_validation_filenames, test_size=0.3)
    random.shuffle(test_filenames)


    # Print results
    print(f'Total point clouds: {len(pts_filenames)}')
    print(f'- {len(train_filenames)} train point clouds')
    print(f'- {len(validation_filenames)} validation point clouds')
    print(f'- {len(test_filenames)} test point clouds')


    # Put results in a dictionary
    output_dict = {
        'train_filenames': train_filenames,
        'validation_filenames': validation_filenames,
        'test_filenames': test_filenames
    }


    # Save dictionary as a JSON file
    json_object = json.dumps(output_dict, indent=2)
    with open('PointCloud_Learning/dataset_filenames_off.json', 'w') as f:
        f.write(json_object)


if __name__ == '__main__':
    splitDataset()
