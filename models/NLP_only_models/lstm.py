import math
from collections import defaultdict

import numpy as np
import torch
import scipy.stats as st
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, confusion_matrix, r2_score, recall_score, f1_score, precision_score
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

    def get_accuracy(self, X, y):
        """
        Get the accuracy of the model on some test set
        :param X: a list of 2d tensors of shape (len(history), input_dim), where each is a single user history sequence
        :param y: a tensor of class labels (1 or 0)
        :return: a float, the accuracy (number of correct predictions out of total)
        """

        y.to(self.device)  # send to gpu if available (X_i are sent later)

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
