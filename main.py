""" playing around with sklearn mlp with mnist digits """

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
# model persistence lib
from sklearn.externals import joblib

IMG_SIZE_BYTES = 784

NUM_TRAINING_SAMPLES = 60000

MODEL_FILE_NAME = 'mlp_digits.pkl'


def train(model, data, labels):
    model.fit(data, labels)
    joblib.dump(model, MODEL_FILE_NAME)
    print("Training set score: %f" % model.score(data, labels))


def load_model():
    model = joblib.load(MODEL_FILE_NAME)
    return model


def predict(img, model):
    prd = model.predict(img)
    prd_prob = model.predict_proba(img)
    print('prediction: ', prd)
    print('prediction probability: ', np.around(prd_prob, decimals=2))
    print('prediction classes: ', model.classes_)


def plot_img(img):
    plt.subplot(111)
    plt.imshow(img.reshape((8, 8)), cmap='gray')
    plt.show()


def load_data():
    # load the labels
    labels = np.loadtxt('labels.txt')[:NUM_TRAINING_SAMPLES]

    # create a large enough array to hold the images
    data = np.zeros((NUM_TRAINING_SAMPLES, IMG_SIZE_BYTES))

    # load the images
    for i in range(1, NUM_TRAINING_SAMPLES):
        data[i - 1, :] = cv2.imread('./images/%s.png' % i, cv2.IMREAD_GRAYSCALE).flatten()/255

    return data, labels


# init the classifier
mlp = MLPClassifier(hidden_layer_sizes=(1000, 1000), max_iter=1000, alpha=1e-5,
                    solver='sgd', tol=1e-4, random_state=1)

# load data from disk
imgs, lbls = load_data()

train(mlp, imgs, lbls)
# mlp = load_model()

# load my image
my_img = cv2.imread('one.png', cv2.IMREAD_GRAYSCALE).reshape((1, -1))

# see what the image classifies to
predict(my_img, mlp)

# visualize first layer weights
fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap='gray', vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()
