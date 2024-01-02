#!/usr/bin/env python3


import os
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from PIL import Image
import open3d as o3d


class Dataset(torch.utils.data.Dataset):

    def __init__(self, filenames):
        self.filenames = filenames
        self.number_of_files = len(self.filenames)

        # Compute the corresponding labels
        self.labels = []
        for filename in self.filenames:
            basename = os.path.basename(filename)
            blocks = basename.split('_')
            label = blocks[0]  # because basename is "bowl_1_1_1.pcd"

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

        pre_transform = T.NormalizeScale()
        self.transforms = T.SamplePoints(1000)

    def __len__(self):
        # must return the size of the data
        return self.number_of_files

    def __getitem__(self, index):
        # Must return the data of the corresponding index

        # Load the image in pil format
        filename = self.filenames[index]
        pcd_object = o3d.io.read_point_cloud(filename)

        # Convert to tensor
        tensor_pcd = self.transforms(pcd_object)

        # Get corresponding label
        label = self.labels[index]

        return tensor_pcd, label
