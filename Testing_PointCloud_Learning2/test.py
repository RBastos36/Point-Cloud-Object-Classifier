import torch
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

#Example usage:
file_path = 'data/objects_pts/bowl_1_1_1.pts'
node_features = read_node_features(file_path)

graph_data = create_graph_data(node_features)

print(graph_data)