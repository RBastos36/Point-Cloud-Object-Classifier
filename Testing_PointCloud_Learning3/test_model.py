from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np
import torch
from torch.utils.data import DataLoader
from classes import PointNet, PointCloudData, default_transforms
import json


print("Beginning testing...")


with open('dataset_filenames_off.json', 'r') as f:
        dataset_filenames = json.load(f)

test_filenames = dataset_filenames['test_filenames']
test_filenames = test_filenames[0:200]                      # NOTE: Change test file number to increase performance time

test_batch_size=len(test_filenames)

test_ds = PointCloudData(valid=True, filenames=test_filenames, transform=default_transforms)
test_loader = DataLoader(dataset=test_ds, batch_size=test_batch_size)


classes = {"bowl": 0,
           "cap": 1,
           "cereal": 2,
           "coffee": 3,
           "soda": 4}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pointnet = PointNet()
pointnet.to(device)
pointnet.load_state_dict(torch.load('save.pth'))


# pointnet.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        print('Batch [%4d / %4d]' % (i+1, len(test_loader)))
        
        inputs, labels = data['pointcloud'].float(), data['category']
        outputs, __, __ = pointnet(inputs.transpose(1,2))
        _, preds = torch.max(outputs.data, 1)
        all_preds += list(preds.numpy())
        all_labels += list(labels.numpy())



cm = confusion_matrix(all_labels, all_preds)



# function from https://deeplizard.com/learn/video/0LhiS6yu2qQ
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plt.figure(figsize=(8,8))
plot_confusion_matrix(cm, list(classes.keys()), normalize=True)
plt.show()



plt.figure(figsize=(8,8))
plot_confusion_matrix(cm, list(classes.keys()), normalize=False)
plt.show()