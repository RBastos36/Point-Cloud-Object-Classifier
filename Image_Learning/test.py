#!/usr/bin/env python3


import json
from dataset import Dataset
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from model import Model

import torch.nn.functional as F


def metrics_params(labels_gt, predicted_is, i):
    # Count FP, FN, TP
    TP, FP, FN = 0, 0, 0
    for gt, pred in zip(labels_gt, predicted_is):

        if gt == i and pred == i:  # True positive
            TP += 1
        elif gt != i and pred == i:  # False positive
            FP += 1
        elif gt == i and pred != i:  # False negative
            FN += 1

    return TP, FP, FN


def metrics_calcs(TP, FP, FN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


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
    with open('Image_Learning/dataset_filenames.json', 'r') as f:
        # Reading from json file
        dataset_filenames = json.load(f)

    test_filenames = dataset_filenames['test_filenames']
    test_filenames = test_filenames[0:2000]

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


    #------------------------------------------------------------
    # Metrics - https://www.evidentlyai.com/classification-metrics/multi-class-metrics
    #------------------------------------------------------------
    TP0, FP0, FN0 = metrics_params(labels_gt_np, predicted_is, 0)   # Bowl metrics
    TP1, FP1, FN1 = metrics_params(labels_gt_np, predicted_is, 1)   # Cap metrics
    TP2, FP2, FN2 = metrics_params(labels_gt_np, predicted_is, 2)   # Cereal Box metrics
    TP3, FP3, FN3 = metrics_params(labels_gt_np, predicted_is, 3)   # Coffee Mug metrics
    TP4, FP4, FN4 = metrics_params(labels_gt_np, predicted_is, 4)   # Soda Can metrics

    precision0, recall0, f1_score0 = metrics_calcs(TP0, FP0, FN0)   # Bowl metrics
    precision1, recall1, f1_score1 = metrics_calcs(TP1, FP1, FN1)   # Cap metrics
    precision2, recall2, f1_score2 = metrics_calcs(TP2, FP2, FN2)   # Cereal Box metrics
    precision3, recall3, f1_score3 = metrics_calcs(TP3, FP3, FN3)   # Coffee Mug metrics
    precision4, recall4, f1_score4 = metrics_calcs(TP4, FP4, FN4)   # Soda Can metrics

    precision_macro = (precision0 + precision1 + precision2 + precision3 + precision4) / 5
    recall_macro = (recall0 + recall1 + recall2 + recall3 + recall4) / 5
    f1_score_macro = 2 * (precision_macro * recall_macro) / (precision_macro + recall_macro)

    precision_micro = (TP0 + TP1 + TP2 + TP3 + TP4) / (TP0 + FP0 + TP1 + FP1 + TP2 + FP2 + TP3 + FP3 + TP4 + FP4)
    recall_micro = (TP0 + TP1 + TP2 + TP3 + TP4) / (TP0 + FN0 + TP1 + FN1 + TP2 + FN2 + TP3 + FN3 + TP4 + FN4)
    f1_score_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro)

    print('Macro Precision = ' + str(precision_macro))
    print('Macro Recall = ' + str(recall_macro))
    print('Macro F1 score = ' + str(f1_score_macro))
    print('Micro Precision = ' + str(precision_micro))
    print('Micro Recall = ' + str(recall_micro))
    print('Micro F1 score = ' + str(f1_score_micro))

    fig = plt.figure()
    idx_image = 0
    for row in range(4):
        for col in range(4):
            image_tensor = inputs[idx_image, :, :, :]
            image_pil = tensor_to_pil_image(image_tensor)
            # print('ground_truth is dog = ' + str(labels_gt_np[idx_image]))
            # print('predicted is dog = ' + str(predicted_is[idx_image]))

            ax = fig.add_subplot(4, 4, idx_image+1)
            plt.imshow(image_pil)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

            text = 'GT: ' + str(variables[labels_gt_np[idx_image]]) + '\nPred: ' + str(variables[predicted_is[idx_image]])

            if labels_gt_np[idx_image] == predicted_is[idx_image]:
                color = 'green'
            else:
                color = 'red'

            ax.set_xlabel(text, color=color)

            idx_image += 1

    plt.show()


if __name__ == "__main__":
    main()
