#!/usr/bin/env python3


import glob
import json
from sklearn.model_selection import train_test_split
from dataset import Dataset
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from model import Model
from trainer import Trainer

import torch.nn.functional as F


def main():

    # -----------------------------------------------------------------
    # Hyperparameters initialization
    # -----------------------------------------------------------------
    learning_rate = 0.001
    num_epochs = 50

    # -----------------------------------------------------------------
    # Create model
    # -----------------------------------------------------------------
    model = Model()

    # -----------------------------------------------------------------
    # Prepare Datasets
    # -----------------------------------------------------------------
    with open('dataset_filenames.json', 'r') as f:
        # Reading from json file
        dataset_filenames = json.load(f)

    test_filenames = dataset_filenames['test_filenames']
    test_filenames = test_filenames[0:1000]

    print('Used ' + str(len(test_filenames)) + ' for testing ')

    test_dataset = Dataset(test_filenames)

    batch_size = len(test_filenames)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # Just for testing the train_loader
    tensor_to_pil_image = transforms.ToPILImage()

    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


    # Load the trained model
    checkpoint = torch.load('models/checkpoint.pkl')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()  # we are in testing mode
    batch_losses = []
    for batch_idx, (inputs, labels_gt) in enumerate(test_loader):

        # move tensors to device
        inputs = inputs.to(device)
        labels_gt = labels_gt.to(device)

        # Get predicted labels
        labels_predicted = model.forward(inputs)

    # Transform predicted labels into probabilities
    predicted_probabilities = F.softmax(labels_predicted, dim=1).tolist()
    # print(predicted_probabilities)

    # probabilities_bowl = [x[0] for x in predicted_probabilities]
    # probabilities_cap = [x[1] for x in predicted_probabilities]
    # probabilities_cereal = [x[2] for x in predicted_probabilities]
    # probabilities_coffee = [x[3] for x in predicted_probabilities]
    # probabilities_soda = [x[4] for x in predicted_probabilities]

    # print(probabilities_dog)

    predicted_is = []
    variables = ["bowl", "cap", "cereal", "coffee", "soda"]
    # Make a decision using the largest probability
    for i in predicted_probabilities:
        idx = i.index(max(i))
        # predicted_is.append(variables[idx])
        predicted_is.append(idx)


    labels_gt_np = labels_gt.cpu().detach().numpy()
    print(labels_gt_np)
    print(predicted_is)
    # ground_truth_is = [x == 0 for x in labels_gt_np]

    # labels_predicted_np = labels_predicted.cpu().detach().numpy()
    # print('labels_gt_np = ' + str(labels_gt_np))
    # print('labels_predicted_np = ' + str(labels_predicted_np))

    # Count FP, FN, TP, and TN
    TP0, FP0, FN0 = 0, 0, 0
    for gt, pred in zip(labels_gt_np, predicted_is):

        if gt == 0 and pred == 0:  # True positive
            TP0 += 1
        elif gt != 0 and pred == 0:  # False positive
            FP0 += 1
        elif gt == 0 and pred != 0:  # False negative
            FN0 += 1

    print('TP = ' + str(TP0))
    print('FP = ' + str(FP0))
    print('FN = ' + str(FN0))

    # Compute precision and recall
    precision = TP0 / (TP0 + FP0)
    recall = TP0 / (TP0 + FN0)
    f1_score = 2 * (precision*recall)/(precision+recall)

    print('Precision = ' + str(precision))
    print('Recall = ' + str(recall))
    print('F1 score = ' + str(f1_score))

    # Show image
    # inputs = inputs.cpu().detach()
    # print(inputs)

    fig = plt.figure()
    idx_image = 0
    for row in range(4):
        for col in range(4):
            image_tensor = inputs[idx_image, :, :, :]
            image_pil = tensor_to_pil_image(image_tensor)
            print('ground_truth is dog = ' + str(labels_gt_np[idx_image]))
            print('predicted is dog = ' + str(predicted_is[idx_image]))

            ax = fig.add_subplot(4, 4, idx_image+1)
            plt.imshow(image_pil)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

            text = 'GT ' + str(labels_gt_np[idx_image]) + '\nPred ' + str(variables[predicted_is[idx_image]])

            if labels_gt_np[idx_image] == predicted_is[idx_image]:
                color = 'green'
            else:
                color = 'red'

            ax.set_xlabel(text, color=color)

            idx_image += 1

    plt.show()


if __name__ == "__main__":
    main()
