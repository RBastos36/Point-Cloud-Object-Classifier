from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np
import torch
from torch.utils.data import DataLoader
from classes import PointNet, PointCloudData, default_transforms
import glob


print("Beginning Object (from Scene) Classification ...")


classification_filenames = glob.glob('Part2_Test/Objects_off/*.off', recursive=True)

classification_batch_size=len(classification_filenames)

classification_ds = PointCloudData(valid=True, filenames=classification_filenames, transform=default_transforms)
classification_loader = DataLoader(dataset=classification_ds, batch_size=classification_batch_size)


classes = {"bowl": 0,
           "cap": 1,
           "cereal": 2,
           "coffee": 3,
           "soda": 4}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pointnet = PointNet()
pointnet.to(device)
pointnet.load_state_dict(torch.load('models/save.pth'))


pointnet.eval()
all_preds = []
with torch.no_grad():
    for i, data in enumerate(classification_loader):
        print('Batch [%4d / %4d]' % (i+1, len(classification_loader)))
        
        inputs, _ = data['pointcloud'].float(), data['category']
        outputs, __, __ = pointnet(inputs.transpose(1,2))
        _, preds = torch.max(outputs.data, 1)
        all_preds += list(preds.numpy())


print("\nPredicted objects:\n")
for i, _ in enumerate(classification_filenames):
    print(str(i+1) + ": " + list(classes.keys())[list(classes.values()).index(all_preds[i])] + "\n")
