import gzip
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import utils

np.random.seed(42)

class LoadMNIST(Dataset):
    def __init__(self, train=False):
        super().__init__()
        if train:
            self.images_path = './data/train-images-idx3-ubyte.gz'
            self.labels_path = './data/train-labels-idx1-ubyte.gz'
        else:
            self.images_path = './data/t10k-images-idx3-ubyte.gz'
            self.labels_path = './data/t10k-labels-idx1-ubyte.gz'

        self.images = self.load_images(self.images_path)
        self.labels = self.load_labels(self.labels_path)

    def __len__(self):
        '''
        method override to compute the length of the data
        '''
        return len(self.images)

    def __getitem__(self, idx):
        '''
        method override to return image tensor and its corresponding label
        '''
        image = self.images[idx]
        label = self.labels[idx]

        return image, label

    def load_images(self, filename):
        '''
        Method to load the train/test images
        :param filename: path to the train/test images file
        :return: images - np.array of shape (-1, 1, 28, 28)
        '''
        if not os.path.exists(filename):
            return 0

        image_size = 28
        f = gzip.open(filename,'r')
        f.read(16)
        buf = f.read()
        images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        # reshape the ndarray to (n_samples, 1, 28, 28) and
        # convert to torch.tensor
        images = images.reshape(-1, 1, image_size, image_size)
        images = torch.from_numpy(images)

        return images

    def load_labels(self, filename):
        '''
        Method to load the train/test labels
        :param filename: path to the train/test labels file
        :return labels: vector of labels
        '''
        if not os.path.exists(filename):
            return 0

        f = gzip.open(filename,'r')
        f.read(8)

        buf = f.read()
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        # convert to torch.tensor and then to one-hot encodeing
        #labels = torch.from_numpy(labels)
        #labels = torch.nn.functional.one_hot(labels)

        return labels

    def show_images(self):
        '''
        Method to display a random images
        :param: none
        :return: none
        '''
        grid_size = 4
        rand_imgs = np.random.choice(len(self), grid_size)

        images = [self.images[i] for i in rand_imgs]
        labels = [int(np.argmax(self.labels[i])) for i in rand_imgs]

        f, axarr = plt.subplots(2,2)
        axarr[0,0].imshow(images[0].numpy(), cmap="gray")
        axarr[0,1].imshow(images[1].numpy(), cmap="gray")
        axarr[1,0].imshow(images[2].numpy(), cmap="gray")
        axarr[1,1].imshow(images[3].numpy(), cmap="gray")

        f.suptitle("labels: " + str(labels))
        plt.show()
