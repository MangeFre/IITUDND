from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import StepLR
import random
import matplotlib.pyplot as plt

# constants
LEARNING_RATE = 0.01
MOMENTUM = 0.9
DECAY_FACTOR = 0.5
EPOCHS = 2

RANDOM_SEED = 42

class LSTM(nn.Module):
    """
    An LSTM to classify tweets in a sequence based on already extracted feature vectors
    Following tutorial on LSTMs found here: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
    """

    def __init__(self, input_dim, hidden_dim):
        """
        build an LSTM
        :param input_dim: the dimensionality of the feature vectors to be input
        :param hidden_dim: the number of neurons used in the LSTM layer
        """
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)      # lstm layer
        self.hidden2class = nn.Linear(hidden_dim, 1)    # fully connected layer

        # set up optimization
        self.loss_function = torch.nn.BCELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=DECAY_FACTOR) # this decreases learning rate every epoch

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

    def learn(self, X, y):
        """
        Train the network using a list of sequences of features and their respective labels
        :param X: a list of 2d tensors of shape (len(history), input_dim), where each is a single user history sequence
        :param y: a 1d tensor of class labels (1 or 0)
        """

        for epoch in range(EPOCHS):

            X, y = shuffle_data(X, y) # shuffle the data each epoch

            print('epoch:', epoch, 'learning rate:', self.scheduler.get_lr())
            running_loss = 0.0 # this variable just for visualization

            for i, X_i in enumerate(X):
                self.zero_grad() # reset the auto gradient calculations

                pred = self(X_i) # forward pass

                # just examine last prediction #todo examine all labeled, not just the last
                loss = self.loss_function(pred[-1], y[i])

                # back propagation
                loss.backward()
                self.optimizer.step()

                # report the running loss on each set of 200 for visualization
                running_loss += loss.item()
                if i % 200 == 199:  # print every 200 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 200))
                    running_loss = 0.0

            # halve learning rate
            self.scheduler.step()
    '''
    def learn_bootstrap(self, X, y, y_bootstrap):
        """
        Train the network using a list of sequences of features and their respective labels
        :param X: a list of 2d tensors of shape (len(history), input_dim), where each is a single user history sequence
        :param y: a tensor of class labels (1 or 0)
        """
        
        #under development
        
        for epoch in range(EPOCHS):

            X, y = shuffle_data(X, y)

            print('epoch:', epoch, 'learning rate:', self.scheduler.get_lr())
            running_loss = 0.0
            for i, X_i in enumerate(X):
                self.zero_grad()
                pred = self(X_i)

                # todo just use target
                loss = self.loss_function(pred[0][0], y[i]) # todo understand what dif is between Size([1]) and Size([])
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 200 == 199:  # print every 200 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 200))
                    running_loss = 0.0

            # halve learning rate
            self.scheduler.step()
            
    '''

    def get_accuracy_graph(self, X, y):
        """
            Get the accuracy of the model on some test set
            :param X: a list of 2d tensors of shape (len(history), input_dim), where each is a single user history sequence
            :param y: a tensor of class labels (1 or 0)
            :return: a list of tuples, each user history length and its mean accuracy
            """
        accByLength = defaultdict(list)  # dict of lists storing accuracies by length
        totalCases = len(X)

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
                accByLength[length].append(1) if predictions == y[i].item() else accByLength[length].append(0)

        # Discretize lengths into bins:

        binMaxCapacity = totalCases // 4 + 1  # define max bin capacity
        accByBin = defaultdict(list)  # new dict storing individual accuracies per bin
        binNum = 1
        binCount = 0
        binMinMax = defaultdict(list) # store the min and max length in each bin
        binMinMax[0].append(1)
        for length in accByLength:
            for item in accByLength[length]:    # iterate through each classification of the hist length
                binCount += 1
                if binCount >= binMaxCapacity:  # move to next bin if current is at max capacity
                    binMinMax[binCount].append(length) # record maximum length of the bin
                    binNum += 1
                    binCount = 0
                    binMinMax[binCount].append(length) # record the min length of the bin
                accByBin[binNum].append(item)  # append the classification value to the bin
        binMinMax[3][1] = length               # record length of final bin

        plt.figure()  # initiate accuracy plot
        bins = []
        accuracy = []

        for bin in accByBin:
            bins.append(bin)
            accuracy.append(np.mean(accByBin[bin]))
        plt.bar(bins, accuracy)  # plot accuracy by history length
        plt.xticks(bins, (binMinMax[0][0] + ' to ' + binMinMax[0][1], binMinMax[1][0] + ' to ' + binMinMax[1][1],
                          binMinMax[2][0] + ' to ' + binMinMax[2][1], binMinMax[3][0] + ' to ' + binMinMax[3][1]))
        plt.suptitle('Test classification accuracy rate by user history length, discretized into four bins')
        plt.xlabel('User history length, discretized into bins (ascending order')
        plt.ylabel('Average accuracy rate')
        plt.show()

        binRatios = []  # compute ratio of true (+1) vs. false (0) classifications
        for bin in accByBin:
            binRatios.append(sum(accByBin[bin]) / len(accByBin[bin]))  # ratio: sum of +1s by total len (+1s and 0s)
        return binRatios

    def get_accuracy(self, X, y):
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
                outputs = self(X_i)  # output contains labels for the whole sequence
                predictions = torch.round(outputs[-1]).item()  # we only care about the last one
                total += 1
                correct += 1 if predictions == y[i].item() else 0
        return correct / total


    '''def get_auc(self,X_test, y_test):
        """
        Get the Area under the ROC curve for some test set
        :param X_test: a tensor of features
        :param y_test: a tensor of labels
        :return: a float, the AUC score
        """
        # make dataloader
        testset = utils.TensorDataset(X_test)  # create your dataset
        testloader = utils.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

        # test model
        y_scores = []
        with torch.no_grad():
            for data in testloader:
                inputs = data[0].to(self.device)
                outputs = self(inputs)
                y_scores.extend(outputs.reshape(-1).tolist())

        return roc_auc_score(y_test.numpy(), np.array(y_scores))
    '''


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
