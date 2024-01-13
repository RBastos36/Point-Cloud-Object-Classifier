from matplotlib import pyplot as plt
from numpy import mean
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import json

try:
    from classes import *
except ImportError:
    from .classes import *


def trainModel(model_path, load_model=False):

    def train(model, train_loader, val_loader=None,  epochs=5):
        
        # Setup matplotlib figure
        plt.figure(num='Training and Validating')
        plt.title('Training and Validation Loss', fontweight="bold")
        plt.axis([1, epochs, 0, 2])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        def draw(epoch_train_losses, epoch_validation_losses):
            xs = range(1, len(epoch_train_losses)+1)
            ys = epoch_train_losses
            ys_validation = epoch_validation_losses

            plt.plot(xs, ys, '-b')
            plt.plot(xs, ys_validation, '-r')
            plt.legend(['Training loss', 'Validation loss'])

            # Draw figure
            plt.draw()
            pressed_key = plt.waitforbuttonpress(1.0) # 1 second
            if pressed_key:
                plt.close()
                raise SystemExit

        epoch_train_losses = []
        epoch_validation_losses = []

        for epoch in range(epochs):
            draw(epoch_train_losses, epoch_validation_losses)
            pointnet.train()
            batch_losses = []
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                optimizer.zero_grad()
                outputs, m3x3, m64x64 = pointnet(inputs.transpose(1,2))

                loss = pointnetloss(outputs, labels, m3x3, m64x64)
                loss.backward()
                optimizer.step()

                # print statistics
                batch_losses.append(loss.data.item())
                print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                    (epoch + 1, i + 1, len(train_loader), loss))

            pointnet.eval()
            correct = total = 0
            epoch_train_loss = mean(batch_losses)
            epoch_train_losses.append(epoch_train_loss)

            # validation
            if val_loader:
                batch_losses = []
                with torch.no_grad():
                    for data in val_loader:
                        inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                        outputs, m3x3, m64x64 = pointnet(inputs.transpose(1,2))
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        validation_loss = pointnetloss(outputs, labels, m3x3, m64x64)
                        batch_losses.append(validation_loss.data.item())

                    epoch_validation_loss = mean(batch_losses)
                    epoch_validation_losses.append(epoch_validation_loss)

                val_acc = 100. * correct / total
                print('Valid accuracy: %d %%' % val_acc)

            # save the model
            torch.save(pointnet.state_dict(), model_path)


    epochs = 10
    train_batch_size = 10 #32
    validation_batch_size = 10 #64


    with open('dataset_filenames_off.json', 'r') as f:
            dataset_filenames = json.load(f)

    train_filenames = dataset_filenames['train_filenames']
    validation_filenames = dataset_filenames['validation_filenames']

    train_filenames = train_filenames[0:50]                # NOTE: Change test file number to increase performance time
    validation_filenames = validation_filenames[0:20]      # NOTE: Change test file number to increase performance time


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
    if load_model:
        print('Load saved model:', model_path)
        pointnet.load_state_dict(torch.load(model_path))


    optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.001)


    train(pointnet, train_loader, valid_loader, epochs)

    plt.show()


if __name__ == '__main__':
    try:
        trainModel(model_path='models/save.pth', load_model=False)
    except SystemExit:
        print('Train stopped by user')
        exit()