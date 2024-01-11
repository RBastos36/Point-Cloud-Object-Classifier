import os
import re
from glob import glob
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchmetrics.classification import MulticlassMatthewsCorrCoef
import open3d as o3
import json

from dataset_pts import Dataset

from open3d.web_visualizer import draw # for non Colab

import matplotlib as mpl
import matplotlib.pyplot as plt

# TEMP for supressing pytorch user warnings
import warnings
warnings.filterwarnings("ignore")

# General parameters
NUM_TRAIN_POINTS = 2500
NUM_TEST_POINTS = 10000
NUM_CLASSES = 5
ROOT = r'data/objects_pts'

# model hyperparameters
GLOBAL_FEATS = 1024

BATCH_SIZE = 32

# get class - label mappings
CATEGORIES = {
    'bowl': 0, 
    'cap': 1, 
    'cereal': 2, 
    'coffee': 3,
    'soda': 4
}
            
# # Simple point cloud coloring mapping for part segmentation
# def read_pointnet_colors(seg_labels):
#     map_label_to_rgb = {
#         1: [0, 255, 0],
#         2: [0, 0, 255],
#         3: [255, 0, 0],
#         4: [255, 0, 255],  # purple
#         5: [0, 255, 255],  # cyan
#         6: [255, 255, 0],  # yellow
#     }
#     colors = np.array([map_label_to_rgb[label] for label in seg_labels])
#     return colors


with open('dataset_filenames_pts.json', 'r') as f:
    dataset_filenames = json.load(f)


train_dataset = ["bowl_1_1_1.txt"]
valid_dataset = dataset_filenames['validation_filenames']
test_dataset = dataset_filenames['test_filenames']


from torch.utils.data import DataLoader

# train Dataset & DataLoader
train_dataloader = DataLoader(dataset=Dataset(train_dataset), batch_size=BATCH_SIZE, shuffle=True)

# # Validation Dataset & DataLoader
# valid_dataloader = DataLoader(dataset=Dataset(valid_dataset), batch_size=BATCH_SIZE)

# # test Dataset & DataLoader 
# test_dataloader = DataLoader(dataset=Dataset(test_dataset), batch_size=BATCH_SIZE)



from point_net import PointNetClassHead

points, targets = next(iter(train_dataloader))

classifier = PointNetClassHead(k=NUM_CLASSES, num_global_feats=GLOBAL_FEATS)
# out, _, _ = classifier(points.transpose(2, 1))
out, _, _ = classifier(((np.array(points)).T).tolist())
# print(f'Class output shape: {out.shape}')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

exit(0)


import torch.optim as optim
from point_net_loss import PointNetLoss

EPOCHS = 100
LR = 0.001
REG_WEIGHT = 0.001 

# use inverse class weighting
# alpha = 1 / class_bins
# alpha = (alpha/alpha.max())

# manually downweight the high frequency classes
alpha = np.ones(NUM_CLASSES)
alpha[0] = 0.5  # airplane
alpha[4] = 0.5  # chair
alpha[-1] = 0.5 # table

gamma = 2

optimizer = optim.Adam(classifier.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, 
                                              step_size_up=2000, cycle_momentum=False)
criterion = PointNetLoss(alpha=alpha, gamma=gamma, reg_weight=REG_WEIGHT).to(DEVICE)

classifier = classifier.to(DEVICE)


mcc_metric = MulticlassMatthewsCorrCoef(num_classes=NUM_CLASSES).to(DEVICE)



def train_test(classifier, dataloader, num_batch, epoch, split='train'):
    ''' Function to train or test the model '''
    _loss = []
    _accuracy = []
    _mcc = []

    # return total targets and predictions for test case only
    total_test_targets = []
    total_test_preds = [] 
    for i, (points, targets) in enumerate(dataloader, 0):

        points = points.transpose(2, 1).to(DEVICE)
        targets = targets.squeeze().to(DEVICE)
        
        # zero gradients
        optimizer.zero_grad()
        
        # get predicted class logits
        preds, _, A = classifier(points)

        # get loss and perform backprop
        loss = criterion(preds, targets, A) 

        if split == 'train':
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # get class predictions
        pred_choice = torch.softmax(preds, dim=1).argmax(dim=1) 
        correct = pred_choice.eq(targets.data).cpu().sum()
        accuracy = correct.item()/float(BATCH_SIZE)
        mcc = mcc_metric(preds, targets)

        # update epoch loss and accuracy
        _loss.append(loss.item())
        _accuracy.append(accuracy)
        _mcc.append(mcc.item())

        # add to total targets/preds
        if split == 'test':
            total_test_targets += targets.reshape(-1).cpu().numpy().tolist()
            total_test_preds += pred_choice.reshape(-1).cpu().numpy().tolist()

        if i % 100 == 0:
            print(f'\t [{epoch}: {i}/{num_batch}] ' \
                  + f'{split} loss: {loss.item():.4f} ' \
                  + f'accuracy: {accuracy:.4f} mcc: {mcc:.4f}')
        
    epoch_loss = np.mean(_loss)
    epoch_accuracy = np.mean(_accuracy)
    epoch_mcc = np.mean(_mcc)

    print(f'Epoch: {epoch} - {split} Loss: {epoch_loss:.4f} ' \
          + f'- {split} Accuracy: {epoch_accuracy:.4f} ' \
          + f'- {split} MCC: {epoch_mcc:.4f}')

    if split == 'test':
        return epoch_loss, epoch_accuracy, epoch_mcc, total_test_targets, total_test_preds
    else: 
        return epoch_loss, epoch_accuracy, epoch_mcc

# stuff for training
num_train_batch = int(np.ceil(len(train_dataset)/BATCH_SIZE))
num_valid_batch = int(np.ceil(len(valid_dataset)/BATCH_SIZE))

# store best validation mcc above 0.
best_mcc = 0.

# lists to store metrics (loss, accuracy, mcc)
train_metrics = []
valid_metrics = []

# TRAIN ON EPOCHS
for epoch in range(1, EPOCHS):

    ## train loop
    classifier = classifier.train()
    
    # train
    _train_metrics = train_test(classifier, train_dataloader, 
                                num_train_batch, epoch, 
                                split='train')
    train_metrics.append(_train_metrics)
        

    # pause to cool down
    time.sleep(4)

    ## validation loop
    with torch.no_grad():

        # place model in evaluation mode
        classifier = classifier.eval()

        # validate
        _valid_metrics = train_test(classifier, valid_dataloader, 
                                    num_valid_batch, epoch, 
                                    split='valid')
        valid_metrics.append(_valid_metrics)

        # pause to cool down
        time.sleep(4)

    # save model if necessary
    if valid_metrics[-1][-1] >= best_mcc:
        best_mcc = valid_metrics[-1][-1]
        torch.save(classifier.state_dict(), 'models/cls_model_%d.pth' % epoch)



metric_names = ['loss', 'accuracy', 'mcc']
_, ax = plt.subplots(len(metric_names), 1, figsize=(8, 6))

for i, m in enumerate(metric_names):
    ax[i].set_title(m)
    ax[i].plot(train_metrics[:, i], label='train')
    ax[i].plot(valid_metrics[:, i], label='valid')
    ax[i].legend()

plt.subplots_adjust(wspace=0., hspace=0.35)
plt.show()
