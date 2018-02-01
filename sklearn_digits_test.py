""" playing around with scikit learn and perceptrons """

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
# model persistence lib
from sklearn.externals import joblib

# Load digits dataset
digits = datasets.load_digits()

# init the classifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=200, alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1)

# Create feature matrix, and target vector
data, labels = digits.data, digits.target


def train():
    mlp.fit(data, labels)
    joblib.dump(mlp, 'mlp_digits.pkl')
    print("Training set score: %f" % mlp.score(data, labels))


def load():
    global mlp
    mlp = joblib.load('mlp_digits.pkl')


def predict():
    prd = mlp.predict(img)
    prd_prob = mlp.predict_proba(img)
    print('prediction: ', prd)
    print('prediction probability: ', np.around(prd_prob, decimals=2))
    print('prediction classes: ', mlp.classes_)


def plot_img():
    plt.subplot(111)
    plt.imshow(img.reshape((8, 8)), cmap='gray')
    plt.show()


img = cv2.imread('my_img.png', cv2.IMREAD_GRAYSCALE).reshape((1, -1)) / 255

train()
# load()
predict()
