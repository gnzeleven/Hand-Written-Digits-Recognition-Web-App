import gzip
import os
import numpy as np
import matplotlib.pyplot as plt

def load_images(filename):
    '''
    Function to load the train/test images
    :param filename: path to the train/test images file
    :return: images - np.array of shape (-1, 28, 28, 1)
    '''
    if not os.path.exists(filename):
        return 0

    image_size = 28
    f = gzip.open(filename,'r')
    f.read(16)
    buf = f.read()
    images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    images = images.reshape(-1, image_size, image_size, 1)

    return images

def load_labels(filename):
    '''
    Function to load the train/test labels
    :param filename: path to the train/test labels file
    :return: images - labels vector of size 60000 for train or 10000 for test
    '''
    if not os.path.exists(filename):
        return 0

    f = gzip.open(filename,'r')
    f.read(8)

    buf = f.read()
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    return labels

def visualize(image):
    '''
    Visualize the image in matplotlib
    :param img: An image of shape (28, 28)
    :return: None
    '''
    if len(image.shape) > 2:
        image = image.reshape(28, 28)
    plt.imshow(image)
    plt.show()

def load_train_test_images():
    '''
    Load the train and test images
    :param: None
    :return train_images: np.array of shape(60000, 28, 28, 1)
    :return test_images: np.array of shape(10000, 28, 28, 1)
    '''
    train_images = load_images('./data/train-images-idx3-ubyte.gz')
    assert (train_images.shape == (60000, 28, 28, 1))
    test_images = load_images('./data/t10k-images-idx3-ubyte.gz')
    assert (test_images.shape == (10000, 28, 28, 1))

    return train_images, test_images

def load_train_test_labels():
    '''
    Load the train and test images
    :param: None
    :return train_labels: np.array of shape(60000,)
    :return test_labels: np.array of shape(10000,)
    '''
    train_labels = load_labels('./data/train-labels-idx1-ubyte.gz')
    assert (train_images.shape == (60000,))
    test_labels = load_labels('./data/t10k-labels-idx1-ubyte.gz')
    assert (test_images.shape == (10000,))

    return train_labels, test_labels


if __name__ == '__main__':
    train_images = load_images('./data/train-images-idx3-ubyte.gz')
    assert (train_images.shape == (60000, 28, 28, 1))
    test_images = load_images('./data/t10k-images-idx3-ubyte.gz')
    assert (test_images.shape == (10000, 28, 28, 1))
    train_labels = load_labels('./data/train-labels-idx1-ubyte.gz')
    assert (train_labels.shape == (60000,))
    test_labels = load_labels('./data/t10k-labels-idx1-ubyte.gz')
    assert (test_labels.shape == (10000,))
    visualize(train_images[2])
