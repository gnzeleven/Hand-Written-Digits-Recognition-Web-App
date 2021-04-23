import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchsummary import summary

from .model import BetterCNN
from .loadmnist import LoadMNIST

def load_model():
    '''
    Function to load the model
    '''
    model = BetterCNN()
    print(summary(model, input_size=(1,28,28)))
    return model

def dataloader(batch_size):
    '''
    Function to create DataLoader objects
    :param batch_size: batch size for train and test
    :return: DataLoader objects trainloader and testloader
    '''
    train_ds = LoadMNIST(train=True)
    trainloader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=True, num_workers=2)

    test_ds = LoadMNIST(train=False)
    testloader = DataLoader(
            test_ds, batch_size=batch_size,
            shuffle=False, num_workers=2)

    return trainloader, testloader

def train(model, trainloader, testloader, n_iters,
            batch_size, learning_rate, optimizer):
    '''
    Function to run the training
    :param n_iters: number of iteration each sample has to run
    :param batch_size: size of the batch
    :param learning_rate: learning rate parameter
    :param optimizer: SGD or Adam
    :return model: trained model
    :return losses: a list of loss computed every 100 iterations
    :return accuracies: a list of accuracy computed every 100 iterations
    '''
    losses = []
    accuracies = []
    num_epochs = int(n_iters / len(trainloader))
    # Error function
    error = nn.CrossEntropyLoss()
    # Optimizer
    if optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # Start training
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(trainloader):

            # Clear gradients
            optimizer.zero_grad()
            # Forward propagation
            logits = model(images)
            # Cross entropy loss
            loss = error(logits, labels)
            # Backprop
            loss.backward()
            # Update parameters
            optimizer.step()

            if i % 100 == 0:
                accuracy = test(model, testloader)
                if epoch == num_epochs - 1:
                    losses.append(loss.data)
                    accuracies.append(accuracy)
                print('Epoch: {} Iteration: {}  Loss: {}  Accuracy: {}%'
                        .format(epoch, i, loss.data, accuracy))

    return model, losses, accuracies

def test(model, testloader):
    '''
    Function to run test
    :param model: Model to compute accuracy
    :param testloader: DataLoader object representing a test batch
    :return: accuracy of the model
    '''
    correct = 0
    total = 0

    for images, labels in testloader:
        logits = model(images)
        # Get predictions from the maximum value
        predicted = torch.max(logits.data, 1)[1]

        # Update total and correct predictions
        total += len(labels)
        correct += (predicted == labels).sum()

    # Compute accuracy
    accuracy = 100 * correct / float(total)

    return accuracy

def plot(losses):
    '''
    function to plot the losses against iteration
    :param losses: a list containing the loss
    :return: none
    '''
    x = [i*100 for i in range(len(losses))]
    plt.plot(x, losses)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Loss vs Iteration')
    plt.savefig('./loss.png')
    plt.show()

def main():
    '''
    Main function - runs the training
    '''
    batch_size = 128
    n_iters = 2500
    learning_rate = 0.001
    optimizer = 'adam'
    PATH = './model/mnist_bcnn.pth'
    model = load_model()
    trainloader, testloader = dataloader(batch_size)
    model, losses, accuracies = train(
                                    model, trainloader, testloader, n_iters,
                                    batch_size, learning_rate, optimizer)
    print("Test accuracy: ", float(accuracies[-1]))
    torch.save(model.state_dict(), PATH)
    print("Model trained and saved...")
    plot(losses)

if __name__ == '__main__':
    main()
