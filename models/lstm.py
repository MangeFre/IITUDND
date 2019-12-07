import math
from collections import defaultdict

import numpy as np
import torch
import scipy.stats as st
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, confusion_matrix, r2_score
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import StepLR
import random
import matplotlib.pyplot as plt

# constants
# LEARNING_RATE = 0.01 # moved to __init__ param default
# MOMENTUM = 0.9 # moved to __init__ param default
# DECAY_FACTOR = 0.5 # moved to __init__ param default
# EPOCHS = 3 # moved to learn param default

RANDOM_SEED = 42

class LSTM(nn.Module):
    """
    An LSTM to classify tweets in a sequence based on already extracted feature vectors
    Following tutorial on LSTMs found here: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
    """

    def __init__(self, input_dim, hidden_dim, num_layers = 1, bidirectional = False,  learning_rate = 0.01, momentum = 0.9, decay_factor = 0.5):
        """
        build an LSTM
        :param input_dim: the dimensionality of the feature vectors to be input
        :param hidden_dim: the number of neurons used in the LSTM layer
        """
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers= num_layers, bidirectional = bidirectional)  # lstm layer
        if bidirectional:
            hidden_dim = hidden_dim * 2
        self.hidden2class = nn.Linear(hidden_dim, 1)    # fully connected layer

        # set up cuda
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # set up optimization
        self.loss_function = torch.nn.BCELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=decay_factor) # this decreases learning rate every epoch

        # set random seed for reproducible data sets
        random.seed(RANDOM_SEED)

    def forward(self, X_i):
        """
        A forward pass of the network to classify each element in the sequence
        :param X_i: a 2d tensor of shape (len(history), input_dim), a single user history sequence
        :return: a 1d tensor of predicted classes (1 for informative, 0 for not)
        """
        input = X_i.view(X_i.shape[0], 1, -1) # need to add a fake dimension for batch (2nd dim out of 3d now)
        lstm_out, _ = self.lstm(input) # lstm_out contains all hidden states for each tweet in the sequence
        fc_in = lstm_out.view(lstm_out.shape[0],-1) # remove fake batch dimension
        preds = torch.sigmoid(self.hidden2class(fc_in)).view(-1) # reduce to 1d tensor after getting scalar predictions
        return preds

    def learn(self, X, y, epochs = 3):
        """
        Train the network using a list of sequences of features and their respective labels
        :param X: a list of 2d tensors of shape (len(history), input_dim), where each is a single user history sequence
        :param y: a 1d tensor of class labels (1 or 0)
        """

        for epoch in range(epochs):

            X, y = shuffle_data(X, y) # shuffle the data each epoch

            print('epoch:', epoch, 'learning rate:', self.scheduler.get_lr())
            #running_loss = 0.0 # this variable just for visualization

            for i, X_i in enumerate(X):
                self.zero_grad() # reset the auto gradient calculations

                pred = self(X_i.to(self.device)) # forward pass

                # just examine last prediction #todo examine all labeled, not just the last
                loss = self.loss_function(pred[-1], y[i].to(self.device))

                # back propagation
                loss.backward()
                self.optimizer.step()

                # report the running loss on each set of 200 for visualization
                #running_loss += loss.item()
                #if i % 200 == 199:  # print every 200 mini-batches
                    #print('[%d, %5d] loss: %.3f' %
                          #(epoch + 1, i + 1, running_loss / 200))
                    #running_loss = 0.0

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
        for binNum in trueByBin: # Iterate through each bin's list of true classifications
            trueVals = np.sum(trueByBin[binNum]) / len(trueByBin[binNum]) # divide + by length of list
            print("True proportion of + scores in bin", binNum, "=", trueVals)
            priors.append(trueVals**2 + (1-trueVals)**2)

        plt.figure()  # initiate accuracy plot
        bins = []
        accuracy = []

        groups = [(str(binMinMax[0][0]) + ' to ' + str(binMinMax[0][1]),  # set the x tick labels
                          str(binMinMax[1][0]) + ' to ' + str(binMinMax[1][1]),
                          str(binMinMax[2][0]) + ' to ' + str(binMinMax[2][1]),
                          str(binMinMax[3][0]) + ' to ' + str(binMinMax[3][1]),
                          str(binMinMax[4][0]) + ' to ' + str(binMinMax[4][1]),
                          str(binMinMax[5][0]) + ' to ' + str(binMinMax[5][1]))]
        for bin in accByBin:
            bins.append(bin)
            accuracy.append(np.mean(accByBin[bin]))
        plt.plot(bins, accuracy, label="Accuracy")  # plot accuracy by bin
        plt.plot(bins, priors, label="Naive accuracy")    # plot dumb accuracy by bin
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

    def get_accuracy(self, X, y):
        """
        Get the accuracy of the model on some test set
        :param X: a list of 2d tensors of shape (len(history), input_dim), where each is a single user history sequence
        :param y: a tensor of class labels (1 or 0)
        :return: a float, the accuracy (number of correct predictions out of total)
        """

        y.to(self.device) # send to gpu if available (X_i are sent later)

        # test model
        correct = 0
        total = 0
        with torch.no_grad():
            for i, X_i in enumerate(X):
                outputs = self(X_i.to(self.device))  # output contains labels for the whole sequence
                predictions = torch.round(outputs[-1]).item()  # we only care about the last one
                total += 1
                correct += 1 if predictions == y[i].item() else 0
        return correct / total


    def get_auc(self,X, y):
        """
        Get the Area under the ROC curve for some test set
        :param X: a list of 2d tensors of shape (len(history), input_dim), where each is a single user history sequence
        :param y: a tensor of class labels (1 or 0)
        :return: a float, the AUC score
        """

        # test model
        y_scores = []
        with torch.no_grad():
            for i, X_i in enumerate(X):
                outputs = self(X_i)  # output contains labels for the whole sequence
                y_scores.append(outputs[-1].item())  # we only care about the last one
        return roc_auc_score(y.numpy(), np.array(y_scores))


    def get_confusion_matrix(self,X_test, y_test):
        """
        Get the confusion matrix of some test set
        :param X_test: a tensor of features
        :param y_test: a tensor of labels
        :return: a confusion matrix
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

    def plot_cis(self, binNames, binRatios):
        '''
        Requires a list of group str outputs and bin ratios from get_accuracy_graph - one for each run
        Collect results of both get_accuracy_plot return values -- names and binRatios-- in an array to run this.
        '''
        names = [bin[0] for bin in binNames] # Establish bin names for the x labels
        binVals = defaultdict(list)
        for run in range(len(binRatios)):
            for bin in range(len(binRatios[run])):
                binVals[bin+1].append(binRatios[run][bin]) # append the ratio (accuracy) of the bin to list
        cis = []
        means = []
        keys = []
        for bin in binVals: # Calculate mean and CI for each bin
            keys.append(bin)
            mean = np.mean(binVals[bin])
            ci = st.t.interval(0.95, len(binVals[bin]) - 1, loc=np.mean(binVals[bin]), scale=st.sem(binVals[bin]))
            cis.append(ci)
            means.append(mean)
        plt.plot(keys, means, label="Mean Accuracy by Bin")  # plot accuracy by bin
        plt.errorbar(keys, means, yerr=cis)
        plt.xticks(keys, names)
        return


def shuffle_data(X, y):
    """
    permute features and labels together
    :param X: a list of 2d tensors of shape (len(history), input_dim), where each is a single user history sequence
    :param y: a tensor of class labels (1 or 0)
    :return: X, y permuted together
    """

    together = list(zip(X, y.tolist()))
    random.shuffle(together)
    X, y = list(zip(*together))

    return X, torch.Tensor(y)
