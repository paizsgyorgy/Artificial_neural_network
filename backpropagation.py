## Import some standard python libraries
import numpy as np
from math import e

## Import the two functions defined in the FeedforwardNN.py file
from FeedforwardNN import sigmoid
from FeedforwardNN import feed_forward_sigmoid
from FeedforwardNN import predict

# ---------------------------------------------------------------------------

## Define the error of prediction
def error(ytrue, ypred):
    return ytrue - ypred

## Define loss function to be used for optimization
def logloss(ytrue, yprob):
    return -(ytrue*np.log(yprob)+(1-ytrue)*np.log(1-yprob))

## Define derivative of sigmoid activation function
def sigmoid_derivative(yhat):
    """Returns the values of sig'(yhat)"""
    sig = sigmoid(yhat)
    return sig*(1-sig)

## Calculate gradient from derivative, logloss function and error as inputs
def gradient(derivative, logloss, error):
    return derivative * logloss * error

## Calculate the change to adjust network weights applied between epochs
def weight_delta(gradient, output_layer, learning_rate):
    return -np.dot(gradient, output_layer) * learning_rate

# ---------------------------------------------------------------------------

## Define backpropagation function to calculate adjusted weights of the network
## This function only runs one epoch of backpropagation

def backpropagation(X, y, w0, w1, LR0=0.01, LR1=0.001):
    """ Runs backpropagation on feed-forward neural network once and returns
        adjusted weights after the epoch."""
    #1) Run network and calculate hidden layer and output layer
    hidden_layer, y_prob = feed_forward_sigmoid(X, w0, w1)
    y_prob = y_prob.flatten()
    y_pred = predict(y_prob)

    #2) Calculate error with logloss function
    lloss = logloss(y, y_prob)

    #3) Calculate error to determine the direction of the gradient
    err = error(y, y_prob)

    #4) Take sigmoid derivative
    sigm_prime = sigmoid_derivative(y_prob)

    #5) Calculate the gradient of the output layer
    grad1 = np.dot(sigm_prime, lloss) * err

    #6) Calculate weights delta for the output layer (w1)
    w1_delta = np.dot(grad1, hidden_layer) * LR1

    #7) Calculate gradient of hidden layer
    X_biased = np.hstack((X, np.ones((X.shape[0],1))))
    grad0 = 1 * grad1 * w1[:2]

    #8) Calculate weights delta for the hidden layer (w2)
    w0_delta = np.dot(grad0, X_biased).T * LR0

    #9) Calculate the updated output layer weights (w1)
    w1 = w1 + w1_delta.reshape(3,1)

    #10) Calculate the updated hidden layer weights (w0)
    w0 = w0 + w0_delta

    return w0, w1

# ---------------------------------------------------------------------------

## Define a training function that runs the learning process of the network

def train_ffnn(X, y, w0, w1, LR0=0.01, LR1=0.001, epochs=500):
    """ Trains a feed forward neural network with one hidden layer.
        Requires X and y as labelled input data and initialized random
        weights, w0 and w1.
        You can optionally specify the learning rate for both layers separately.
        Returns the final trained weights of the model."""
    for i in range(epochs):
        w0, w1 = backpropagation(X, y, w0, w1, LR0, LR1)
    return w0, w1
