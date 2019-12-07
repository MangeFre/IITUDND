import math

import scipy.stats as st
import torch.nn as nn
import torch
import torch.utils.data as utils
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.metrics import roc_auc_score, r2_score, confusion_matrix
import matplotlib.pyplot as plt
from collections import defaultdict
import torch.utils.data as data_utils


# constants
# LEARNING_RATE = 0.01 # moved to __init__ param default
# MOMENTUM = 0.9 # moved to __init__ param default
# DECAY_FACTOR = 0.5 # moved to __init__ param default
# EPOCHS = 3 # moved to learn param default

class MLP(nn.Module):
    """
    A multilayer perceptron based on the starter tutorial at pytorch:
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
    """
    def __init__(self, input_size, hidden_dim, num_layers= 1, activation_function = torch.relu, learning_rate = 0.001, decay_factor = 0.5, momentum = 0.9):
        """
        A simple 4 layer network for binary classification
        :param input_size: an int
        :param hidden_dim: an int
        """

        self.f = activation_function

        super(MLP, self).__init__()

        self.fc_first = nn.Linear(input_size, hidden_dim)
        self.layers = []
        for i in range(num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc_last = nn.Linear(hidden_dim, 1)

        # set up cuda
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        for i in range(len(self.layers)):
            self.layers[i].to(self.device)

        # set up optimization
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=decay_factor)

    def forward(self, x):
        x = self.f(self.fc_first(x))
        for i in range(len(self.layers)):
            x = self.f(self.layers[i](x))
        x = torch.sigmoid(self.fc_last(x))
        return x

    def learn(self, X_train, y_train, epochs = 3):
        """
        Train the network on a labeled dataset
        :param X_train: a tensor of features
        :param y_train: a tensor of labels
        """

        # make dataloader
        trainset = utils.TensorDataset(X_train, y_train)
        trainloader = utils.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)


        # train model
        for epoch in range(epochs):
            print('epoch:', epoch, 'learning rate:', self.scheduler.get_lr())
            running_loss = 0.0
            for j, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs.reshape(-1), labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if j % 200 == 199:  # print every 200 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, j + 1, running_loss / 200))
                    running_loss = 0.0

            # halve learning rate
            self.scheduler.step()

    def get_accuracy_graph(self, X, y):
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

        # test model
        correct = 0
        total = 0
        with torch.no_grad():
            for i, X_i in enumerate(X):
                length = X_i.shape[0]  # user history length
                outputs = self(X_i)  # output contains labels for the whole sequence
                predictions = torch.round(outputs[-1]).item()  # we only care about the last one
                total += 1
                correct += 1 if predictions == y[i].item() else 0
                predByLength[length].append(predictions) # store predicted value (need this to get R2 in bins)
                accByLength[length].append(1) if predictions == y[i].item() else accByLength[length].append(0)
                trueByLength[length].append(y[i].item()) # keep track of original classification
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

        # Calculate priors per bin:
        priors = []
        for binNum in trueByBin:  # Iterate through each bin's list of true classifications
            trueVals = np.sum(trueByBin[binNum]) / len(trueByBin[binNum])  # divide + by length of list
            print("True proportion of + scores in bin", binNum, "=", trueVals)
            priors.append(trueVals ** 2 + (1 - trueVals) ** 2)

        plt.figure()  # initiate accuracy plot
        bins = []
        accuracy = []

        for bin in accByBin:
            bins.append(bin)
            accuracy.append(np.mean(accByBin[bin]))
        plt.plot(bins, accuracy, label="Accuracy")  # plot accuracy by bin
        plt.plot(bins, priors, label="Naive accuracy")  # plot dumb accuracy by bin
        groups = [str(binMinMax[0][0]) + ' to ' + str(binMinMax[0][1]),  # set the x tick labels
                          str(binMinMax[1][0]) + ' to ' + str(binMinMax[1][1]),
                          str(binMinMax[2][0]) + ' to ' + str(binMinMax[2][1]),
                          str(binMinMax[3][0]) + ' to ' + str(binMinMax[3][1]),
                          str(binMinMax[4][0]) + ' to ' + str(binMinMax[4][1]),
                          str(binMinMax[5][0]) + ' to ' + str(binMinMax[5][1])]
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
        return groups, binRatios

    def plot_CIs(self, binNames, binRatios):
        '''
        Requires a list of group str outputs and bin ratios from get_accuracy_graph - one for each run
        Collect results of both get_accuracy_plot return values -- names and binRatios-- in an array to run this.
        '''
        names = [bin[0] for bin in binNames] # Establish bin names for the x labels
        binVals = defaultdict(list)
        for run in binRatios:
            for bin in run:
                binVals[bin].append(binRatios[run][bin])
        cis = []
        means = []
        for bin in binVals: # Calculate mean and CI for each bin
            mean = np.mean(binVals[bin])
            ci = st.t.interval(0.95, len(binVals[bin]) - 1, loc=np.mean(binVals[bin]), scale=st.sem(binVals[bin]))
            cis.append(ci)
            means.append(mean)
        plt.plot(binVals.keys(), means, label="Mean Accuracy by Bin")  # plot accuracy by bin
        plt.errorbar(binVals.keys(), means, yerr=cis)
        plt.xticks(binVals.keys(), names)
        return


    # todo how did this get here and how was it working for so long!
    '''def get_accuracy(self, X, y):
        """
        Get the accuracy of the model on some test set
        :param X: a list of 2d tensors of shape (len(history), input_dim), where each is a single user history sequence
        :param y: a tensor of class labels (1 or 0)
        :return: a float, the accuracy (number of correct predictions out of total)
        """

        # test model
        correct = 0
        total = 0
        with torch.no_grad():
            for i, X_i in enumerate(X):
                outputs = self(X_i)                             # output contains labels for the whole sequence
                predictions = torch.round(outputs[-1]).item()   # we only care about the last one
                total += 1
                correct += 1 if predictions == y[i].item() else 0
        return correct / total'''

    def get_accuracy(self, X_test, y_test):
        """
        Get the accuracy of the model on some test set
        :param X_test: a tensor of features
        :param y_test: a tensor of labels
        :return: a float, the accuracy (number of correct predictions out of total)
        """

        # make dataloader
        testset = utils.TensorDataset(X_test, y_test)  # create your datset
        testloader = utils.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)
        testset = utils.TensorDataset(X_test, y_test)
        testloader = utils.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

        # test model
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self(inputs)
                predictions = torch.round(outputs).reshape(-1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        return correct / total

    def get_auc(self,X_test, y_test):
        """
        Get the Area under the ROC curve for some test set
        :param X_test: a tensor of features
        :param y_test: a tensor of labels
        :return: a float, the AUC score
        """
        # make dataloader
        testset = data_utils.TensorDataset(X_test)  # create your dataset
        testloader = data_utils.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

        # test model
        y_scores = []
        with torch.no_grad():
            for data in testloader:
                inputs = data[0].to(self.device)
                outputs = self(inputs)
                y_scores.extend(outputs.reshape(-1).tolist())

        return roc_auc_score(y_test.numpy(), np.array(y_scores))


    def get_confusion_matrix(self,X_test, y_test):
        """
        Get the confusion matrix of some test set
        :param X_test: a tensor of features
        :param y_test: a tensor of labels
        :return: a float, the AUC score
        """
        # make dataloader
        testset = data_utils.TensorDataset(X_test)  # create your dataset
        testloader = data_utils.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

        # test model
        y_scores = []
        with torch.no_grad():
            for data in testloader:
                inputs = data[0].to(self.device)
                outputs = self(inputs)
                y_scores.extend(outputs.reshape(-1).tolist())

        return confusion_matrix(y_test.numpy(), np.array(y_scores))
