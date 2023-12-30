#!/usr/bin/env python3


import os
import torch
from torchvision import transforms
from PIL import Image


class Dataset(torch.utils.data.Dataset):

    def __init__(self, filenames):
        self.filenames = filenames
        self.number_of_images = len(self.filenames)

        # Compute the corresponding labels
        self.labels = []
        for filename in self.filenames:
            basename = os.path.basename(filename)
            blocks = basename.split('_')
            label = blocks[0]  # because basename is "bowl_1_1_1_crop.png"

            if label == 'bowl':
                self.labels.append(0)
            elif label == 'cap':
                self.labels.append(1)
            elif label == 'cereal':
                self.labels.append(2)
            elif label == 'coffee':
                self.labels.append(3)
            elif label == 'soda':
                self.labels.append(4)
            else:
                raise ValueError('Unknown label ' + label)

        # print(self.filenames[0:3])
        # print(self.labels[0:3])
        # indexes = [0, 1, 2, ...]
        # filenames ['/home/mike/savi_datasets/dogs-vs-cats/train/cat.2832.jpg', '/home/mike/savi_datasets/dogs-vs-cats/train/cat.8274.jpg', '/home/mike/savi_datasets/dogs-vs-cats/train/cat.4537.jpg']
        # labels ['cat', 'cat', 'cat']

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        # must return the size of the data
        return self.number_of_images

    def __getitem__(self, index):
        # Must return the data of the corresponding index

        # Load the image in pil format
        filename = self.filenames[index]
        pil_image = Image.open(filename)

        # Convert to tensor
        tensor_image = self.transforms(pil_image)

        # Get corresponding label
        label = self.labels[index]

        return tensor_image, label
