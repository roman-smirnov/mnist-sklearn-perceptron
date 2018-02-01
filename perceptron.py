""" implementing a simple perceptron """

import numpy as np
import cv2
import matplotlib.pyplot as plt

# constants
IMG_SIZE_BYTES = 784
NUM_TRAINING_SAMPLES = 6000

# load MNIST images and image labels
def load_data():
    # load the labels
    labels = np.loadtxt('labels.txt')[:NUM_TRAINING_SAMPLES]

    # create a large enough array to hold the images
    data = np.zeros((NUM_TRAINING_SAMPLES, IMG_SIZE_BYTES))

    # load the images
    for i in range(1, NUM_TRAINING_SAMPLES):
        data[i - 1, :] = cv2.imread('./images/%s.png' % i, cv2.IMREAD_GRAYSCALE).flatten() / 255

    return data, labels


# load data from disk
# imgs, lbls = load_data()

# class Perceptron:
#     def __init__(self, hidden_layers = 100, max_iter = 100 ):
#        self.

# TODO: implement 