import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import random

# constants
LEARNING_RATE = 0.01
MOMENTUM = 0.9
DECAY_FACTOR = 0.5
EPOCHS = 2

RANDOM_SEED = 42

class LSTM(nn.Module):
    """
    An LSTM to classify the last tweet in a sequence based on feature vectors
    Following tutorial on LSTMs found here: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
    """

    def __init__(self, input_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.hidden2class = nn.Linear(hidden_dim, 1)

        self.loss_function = torch.nn.BCELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=DECAY_FACTOR)

        # set random seed for reproducable data sets
        random.seed(RANDOM_SEED)

    def forward(self, X_i):
        """
        A forward pass of the network to classify the last element in the sequence
        :param X_i: a 2d tensor of shape (len(history), input_dim), a single user history sequence
        :return: a predicted class (1 for informative, 0 for not)
        """
        input = X_i.view(X_i.shape[0], 1, -1) # need to add a dimension for batch (2nd dim out of 3d now)
        lstm_out, _ = self.lstm(input) # lstm_out contains all hidden states for the sequence
        preds = torch.sigmoid(self.hidden2class(lstm_out.view(lstm_out.shape[0],-1))) # reduce to 2d
        return preds

    def learn(self, X, y):
        """
        Train the network using a list of sequences of features and their respective labels
        :param X: a list of 2d tensors of shape (len(history), input_dim), where each is a single user history sequence
        :param y: a tensor of class labels (1 or 0)
        """

        for epoch in range(EPOCHS):

            X, y = shuffle_data(X, y)

            print('epoch:', epoch, 'learning rate:', self.scheduler.get_lr())
            running_loss = 0.0
            for i, X_i in enumerate(X):
                self.zero_grad()
                pred = self(X_i)

                # just examine last prediction #todo examine all labeled
                loss = self.loss_function(pred[-1][0], y[i]) # todo understand what dif is between Size([1]) and Size([])
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
                outputs = self(X_i)
                predictions = torch.round(outputs[-1]).reshape(-1).item()
                total += 1
                correct += 1 if predictions == y[i].item() else 0

        return correct / total



    #TODO IMPLEMeNT
    '''
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