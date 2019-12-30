import math

import scipy.stats as st
import torch.nn as nn
import torch
import torch.utils.data as utils
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.metrics import roc_auc_score, r2_score, confusion_matrix, f1_score, recall_score, precision_score
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
        A simple multi layer network for binary classification
        :param input_size: the dimensionality of the feature vectors to be input
        :param hidden_dim: the number of neurons used in each layers
        :param num_layers: number of layers in the network
        :param activation_function: the nonlinearity used after each layer except the last
        :param learning_rate: coefficient for gradient decent
        :param decay_factor: coefficient for scheduled learning_rate decrease per epoch
        :param momentum: momentum coefficient for gradient decent
        """
        super(MLP, self).__init__()

        self.f = activation_function

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
        """
        A forward pass of the network to classify a given tweet
        :param x: a feature vector representing a tweet
        :return: a 1d tensor of predicted classes (between 1 for informative, 0 for not)
        """
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
        :param epochs: an int, number of epochs to learn for
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

    def get_accuracy(self, X_test, y_test):
        """
        Get the accuracy of the model on some test set
        :param X_test: a tensor of features
        :param y_test: a tensor of labels
        :return: a float, the accuracy (number of correct predictions out of total)
        """

        # make dataloader
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