import numpy as np
from collections import defaultdict
import statistics
import matplotlib.pyplot as plt

import torch.utils.data as utils
from collections import defaultdict
import numpy as np
import torch
import scipy.stats as st
from sklearn.metrics import roc_auc_score, confusion_matrix, r2_score, precision_score, recall_score, f1_score
import torch.utils.data as data_utils

import matplotlib.pyplot as plt


def plot_cis(binNames, binRatios, priors):
    '''
    Requires a list of group str outputs and bin ratios from get_accuracy_graph - one for each run
    Collect results of both get_accuracy_plot return values -- names and binRatios-- in an array to run this.
    '''
    priors = np.array(priors[0])
    binVals = defaultdict(list)
    for run in range(len(binRatios)):
        for bin in range(len(binRatios[run])):
            binVals[bin+1].append(binRatios[run][bin]) # append the ratio (accuracy) of the bin to list
    cis = []
    means = []
    keys = []
    binLabels = [name for name in binNames[0]]
    for bin in binVals: # Calculate mean and CI for each bin
        keys.append(bin)
        mean = np.mean(binVals[bin])
        means.append(mean)
        standard = statistics.stdev(binVals[bin])
        cis.append(standard)
    plt.figure()  # initiate accuracy plot
    plt.plot(keys, means, label="Mean Accuracy by Bin")  # plot accuracy by bin
    plt.plot(keys, priors, label="Naive Accuracy")
    plt.errorbar(keys, means, yerr=cis)
    plt.xticks(keys, binLabels)
    plt.suptitle('Test classification accuracy rate by user history length (CI .95)')
    plt.xlabel('User history length (lowest to highest), sorted into bins (ascending order)')
    plt.ylabel('Accuracy rate')
    plt.show()
    return


# lstm

def get_accuracy_graph(model, X, X_img, y, X_hist_len_test, y_train, is_lstm, is_multimodal=True):
    """
        Get the accuracy of the model on some test set
        :param X: a list of 2d tensors of shape (len(history), input_dim), where each is a single user history
        sequence
        :param y: a tensor of class labels (1 or 0)
        :return: a plot of accuracies across bins of history lengths and a list of each bin's mean accuracy
        """
    accByLength = defaultdict(list)  # dict of lists storing accuracies by length
    totalCases = len(X)

    trueByLength = defaultdict(list)
    predByLength = defaultdict(list)

    if is_lstm:
        y.to(model.device)  # send to gpu if available (X_i are sent later)

        # test model
        correct = 0
        total = 0
        with torch.no_grad():
            for i, X_i in enumerate(X):
                length = X_i.shape[0]  # user history length
                if is_multimodal:
                    X_i_images = X_img[i]
                    outputs = model(X_i.to(model.device), X_i_images)  # output contains labels for the whole sequence
                else:
                    outputs = model(X_i.to(model.device))  # output contains labels for the whole sequence
                predictions = torch.round(outputs[-1]).item()  # we only care about the last one
                total += 1
                correct += 1 if predictions == y[i].item() else 0
                predByLength[length].append(predictions) # store predicted value (need this to get R2 in bins)
                accByLength[length].append(1) if predictions == y[i].item() else accByLength[length].append(0)
                trueByLength[length].append(y[i].item()) # keep track of original classification

    # feedforward
    else:
        # make dataloader

        testset = utils.TensorDataset(X, y)
        testloader = utils.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

        # test model
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                length = X_hist_len_test[i]
                inputs, labels = data[0].to(model.device), data[1].to(model.device)
                outputs = model(inputs)
                predictions = torch.round(outputs).item()
                total += 1
                correct += 1 if predictions == y[i].item() else 0
                predByLength[length].append(predictions)  # store predicted value (need this to get R2 in bins)
                accByLength[length].append(1) if predictions == y[i].item() else accByLength[length].append(0)
                trueByLength[length].append(y[i].item())  # keep track of original classification

    predByLength = sorted(predByLength.items()) # get every dict in order via a sorted list of tuples (len, vals)
    accByLength = sorted(accByLength.items())
    trueByLength = sorted(trueByLength.items())

    # Discretize lengths into bins:
    binMaxCapacity = totalCases // 6 + 1  # define max bin capacity
    accByBin = defaultdict(list)  # new dict storing individual accuracies (1=correct,0=wrong) per bin
    trueByBin = defaultdict(list) # new dict storing individual true values per bin
    predByBin = defaultdict(list) # new dict storing predicted values per bin (need for R2)
    binNum = 0
    binCount = 0
    binMinMax = defaultdict(list) # store the min and max length in each bin
    binMinMax[0].append(1)
    for length in range(len(accByLength)):            # iterate through each hist length
        for i in range(len(accByLength[length][1])):  # iterate through each value in the current hist length
            binCount += 1
            if binCount >= binMaxCapacity:  # move to next bin if current is at max capacity
                binMinMax[binNum].append(accByLength[length][0]) # record maximum length of the bin
                binNum += 1
                binCount = 0
                binMinMax[binNum].append(accByLength[length][0]) # record the min length of the bin
            trueByBin[binNum].append(trueByLength[length][1][i])  # append the true value to the bin list
            accByBin[binNum].append(accByLength[length][1][i])  # append the classification accuracy to the bin list
            predByBin[binNum].append(predByLength[length][1][i]) # append predicted value to bin list
    binMinMax[5].append(accByLength[length][0])                 # record length of final bin

    # Calculate R score: (Turn into separate method)
    for binNum in accByBin:
        predictedVals = predByBin[binNum]
        trueVals = trueByBin[binNum]
        print("R2 score for bin", binNum, "=", r2_score(trueVals, predictedVals))

    # Naive classifier results per bin:
    naiveClassifier = []
    a = np.sum(y_train) / len(y_train)
    for binNum in trueByBin:
        b = (np.sum(trueByBin[binNum]) / len(trueByBin[binNum]))
        naiveClassifier.append((a*b) + (1-a)*(1-b))

    # Calculate priors per bin:
    priors = []
    for binNum in trueByBin: # Iterate through each bin's list of true classifications
        trueVals = np.sum(trueByBin[binNum]) / len(trueByBin[binNum]) # divide + by length of list
        print("True proportion of + scores in bin", binNum, "=", trueVals)
        priors.append(trueVals**2 + (1-trueVals)**2)

    plt.figure()  # initiate accuracy plot
    bins = []
    accuracy = []

    groups = [str(binMinMax[0][0]) + ' to ' + str(binMinMax[0][1]),  # set the x tick labels
                      str(binMinMax[1][0]) + ' to ' + str(binMinMax[1][1]),
                      str(binMinMax[2][0]) + ' to ' + str(binMinMax[2][1]),
                      str(binMinMax[3][0]) + ' to ' + str(binMinMax[3][1]),
                      str(binMinMax[4][0]) + ' to ' + str(binMinMax[4][1]),
                      str(binMinMax[5][0]) + ' to ' + str(binMinMax[5][1])]
    for bin in accByBin:
        bins.append(bin)
        accuracy.append(np.mean(accByBin[bin]))
    plt.plot(bins, accuracy, label="Accuracy")  # plot accuracy by bin
    plt.plot(bins, naiveClassifier, label="Naive classifier accuracy")    # plot dumb accuracy by bin
    plt.xticks(bins, groups)
    plt.suptitle('Test classification accuracy rate by user history length, separated into six bins')
    plt.xlabel('User history length (lowest to highest), discretized into bins (ascending order)')
    plt.ylabel('Average accuracy rate')
    plt.ylim(0.5, 0.9)
    plt.yticks(np.arange(0.5, 0.9, 0.05))
    plt.show()

    ''' Compute ratios of true classifications to false classifications'''
    binRatios = []  # compute ratio of true (+1) vs. false (0) classifications
    for bin in accByBin:
        binRatios.append(sum(accByBin[bin]) / len(accByBin[bin]))  # ratio: sum of +1s by total len (+1s and 0s)
    return groups, binRatios, naiveClassifier


def get_acc_auc_pre_re_f1(model, X, X_img, y, is_lstm, is_multimodal=True):
    """
    Get the recall of some test set
    :param X: a list of 2d tensors of shape (len(history), input_dim), where each is a single user history sequence
    :param y: a tensor of class labels (1 or 0)
    :return: a float, the AUC score
    """
    # test model
    y_preds = []
    y_scores = []
    correct = 0
    total = 0

    if is_lstm:
        with torch.no_grad():
            for i, X_i in enumerate(X):
                if is_multimodal:
                    X_i_images = X_img[i]
                    outputs = model(X_i.to(model.device), X_i_images)
                else:
                    outputs = model(X_i.to(model.device))  # output contains labels for the whole sequence
                y_scores.append(outputs[-1].item())
                predictions = torch.round(outputs[-1]).item()
                y_preds.append(predictions)  # we only care about the last one
                total += 1
                correct += 1 if predictions == y[i].item() else 0

    # feedforward
    else:
        # make dataloader
        testset = data_utils.TensorDataset(X, y)  # create your dataset
        testloader = data_utils.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(model.device), data[1].to(model.device)
                outputs = model(inputs)
                y_scores.extend(outputs.reshape(-1).tolist())
                predictions = torch.round(outputs).reshape(-1)
                y_preds.extend(predictions.tolist())  # we only care about the last one
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

    return correct/total, roc_auc_score(y.numpy(), np.array(y_scores)), \
           precision_score(y.numpy(), np.array(y_preds)), \
           recall_score(y.numpy(), np.array(y_preds)), \
           f1_score(y.numpy(), np.array(y_preds))