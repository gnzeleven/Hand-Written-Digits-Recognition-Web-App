import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
    """CNN class - defines model and forward operations"""

    def __init__(self):
        super(CNN, self).__init__()

        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(
                in_channels=1, out_channels=8,
                kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
                in_channels=8, out_channels=16,
                kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(
                in_channels=16, out_channels=32,
                kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(
                in_channels=32, out_channels=64,
                kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(128, 10)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        '''
        Method override for forward operation
        '''
        # conv1 -> relu -> pooling
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)
        # conv2 -> relu -> pooling
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling(x)
        # conv3 -> relu -> pooling
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pooling(x)
        # conv4 -> relu -> pooling
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pooling(x)
        # flatten
        x = x.view(-1, 64)
        # fully connected -> relu -> dropout
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # fully connected -> softmax
        x = self.fc2(x)
        out = self.logsoftmax(x)

        return out

class BetterCNN(nn.Module):
    """BetterCNN class - same as CNN but better performance"""

    def __init__(self):
        super(BetterCNN, self).__init__()

        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(
                in_channels=1, out_channels=32,
                kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(
                in_channels=32, out_channels=16,
                kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(400, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        '''
        Method override for forward operation
        '''
        # conv1 -> relu -> pooling
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)
        # conv2 -> relu -> pooling
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling(x)
        # flatten
        x = x.view(-1, 400)
        # fully connected -> relu -> dropout
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # fully connected -> softmax
        x = self.fc2(x)
        out = self.logsoftmax(x)

        return out
