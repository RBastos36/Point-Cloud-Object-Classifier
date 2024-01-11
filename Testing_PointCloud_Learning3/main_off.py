import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import json
from classes import *


def train(model, train_loader, val_loader=None,  epochs=5):
    for epoch in range(epochs):
        pointnet.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = pointnet(inputs.transpose(1,2))

            loss = pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:    # print every 10 mini-batches
                print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                    (epoch + 1, i + 1, len(train_loader), running_loss / 10))
                running_loss = 0.0

        pointnet.eval()
        correct = total = 0

        # validation
        if val_loader:
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, __, __ = pointnet(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100. * correct / total
            print('Valid accuracy: %d %%' % val_acc)

        # save the model
        torch.save(pointnet.state_dict(), "save.pth")


epochs = 10
train_batch_size=32
validation_batch_size=64


with open('dataset_filenames_off.json', 'r') as f:
        dataset_filenames = json.load(f)

train_filenames = dataset_filenames['train_filenames']
validation_filenames = dataset_filenames['validation_filenames']

train_filenames = train_filenames[0:500]                # NOTE: Change test file number to increase performance time
validation_filenames = validation_filenames[0:200]      # NOTE: Change test file number to increase performance time


classes = {"bowl": 0,
           "cap": 1,
           "cereal": 2,
           "coffee": 3,
           "soda": 4}


train_transforms = transforms.Compose([
                    PointSampler(1024),
                    Normalize(),
                    RandRotation_z(),
                    RandomNoise(),
                    ToTensor()
                    ])

train_ds = PointCloudData(filenames=train_filenames, transform=train_transforms)
valid_ds = PointCloudData(valid=True, filenames=validation_filenames, transform=train_transforms)

inv_classes = {i: cat for cat, i in train_ds.classes.items()}


print('Train dataset size: ', len(train_ds))
print('Valid dataset size: ', len(valid_ds))
print('Number of classes: ', len(train_ds.classes))
print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())


train_loader = DataLoader(dataset=train_ds, batch_size=train_batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_ds, batch_size=validation_batch_size)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)


pointnet = PointNet()
pointnet.to(device)

# Load a pre-trained model if it exists
# pointnet.load_state_dict(torch.load('save.pth'))


optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.001)


train(pointnet, train_loader, valid_loader, epochs)
