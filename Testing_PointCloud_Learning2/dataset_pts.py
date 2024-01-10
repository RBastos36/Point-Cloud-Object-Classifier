#!/usr/bin/env python3


import os
import torch
from torchvision import transforms
import torch_geometric.transforms as T
from PIL import Image
from torch_geometric.data import Data



def read_node_features(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        node_features = [list(map(float, line.strip().split())) for line in lines]

    return node_features

def create_graph_data(node_features):
    x = torch.tensor(node_features, dtype=torch.float)

    #For simplicity, let's create a graph with no edges.
    edge_index = torch.tensor([], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    return data


class Dataset(torch.utils.data.Dataset):

    def __init__(self, filenames):
        self.filenames = filenames
        self.number_of_images = len(self.filenames)

        # Compute the corresponding labels
        self.labels = []
        for filename in self.filenames:
            basename = os.path.basename(filename)
            print(basename)
            blocks = basename.split('_')
            label = blocks[0]  # because basename is "bowl_1_1_1.pts"

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

        self.pre_transforms = T.NormalizeScale()
        self.transforms = T.SamplePoints(1024)

        # self.transforms = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor()
        # ])

    def __len__(self):
        # must return the size of the data
        return self.number_of_images

    def __getitem__(self, index):
        # Must return the data of the corresponding index

        # Load the image in pil format
        filename = self.filenames[index]

        # Convert to tensor
        # pre_tensor_pointcloud = self.pre_transforms(filename)
        node_features = read_node_features(filename)
        graph_data = create_graph_data(node_features)
        tensor_pointcloud = self.transforms(graph_data)

        # Get corresponding label
        label = self.labels[index]

        return tensor_pointcloud, label
    
