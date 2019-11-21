import torch.nn as nn
import torch
import torch.utils.data as utils
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.metrics import roc_auc_score


# constants
LEARNING_RATE = 0.1
DECAY_FACTOR = 0.1
EPOCHS = 2

class MLP(nn.Module):
    """
    A multilayer perceptron based on the starter tutorial at pytorch:
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
    """
    def __init__(self, input_size, first_layer_size, second_layer_size, activation_function):
        """
        A simple 4 layer network for binary classification
        :param input_size: an int
        :param first_layer_size: an int
        :param second_layer_size: an int
        :param activation_function: a function, an activation function for all neurons except last layer
        """

        self.f = activation_function

        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, first_layer_size)
        self.fc2 = nn.Linear(first_layer_size, second_layer_size)
        self.fc3 = nn.Linear(second_layer_size, 1)

        # set up cuda
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # set up optimization
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=LEARNING_RATE)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=DECAY_FACTOR)

    def forward(self, x):
        x = self.f(self.fc1(x))
        x = self.f(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def learn(self, X_train, y_train):
        """
        Train the network on a labeled dataset
        :param X_train: a tensor of features
        :param y_train: a tensor of labels
        """

        # make dataloader
        trainset = utils.TensorDataset(X_train, y_train)
        trainloader = utils.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)


        # train model
        for epoch in range(EPOCHS):
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

    def get_auc(self,X_test, y_test):
        """
        Get the Area under the ROC curve for some test set
        :param X_test: a tensor of features
        :param y_test: a tensor of labels
        :return: a float, the AUC score
        """
        # make dataloader
        testset = utils.TensorDataset(X_test)  # create your datset
        testloader = utils.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

        # test model
        y_scores = []
        with torch.no_grad():
            for data in testloader:
                inputs = data[0].to(self.device)
                outputs = self(inputs)
                y_scores.extend(outputs.reshape(-1).tolist())

        return roc_auc_score(y_test.numpy(), np.array(y_scores))
