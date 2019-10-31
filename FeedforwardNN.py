## Import all necessary packages
import numpy as np
from math import e

# ---------------------------------------------------------------------------

## Define sigmoid activation function
def sigmoid(x):
    """ Returns result of a sigmoid functional call on the input number x."""
    return 1 / (1 + e ** -x)

# ---------------------------------------------------------------------------

## Define function to predict results from output layer probabilities
## Note: this only works with binary classification
def predict(o2):
    return np.array([np.round(x) for x in o2])

# ---------------------------------------------------------------------------

## Define function to calculate feed-forward neural network
def feed_forward_sigmoid(X, w0, w1):
    """ Returns the hidden and the outer layer of a feed forward
        neural network with sigmoid activation function.
        X: input data for the neural network with shape (50,2)"""
    ## 1: Check that X data has shape (50, 2)
    assert X.shape == (50,2)

    ## 2: add a bias columns, so X turns from (50,2) to (50,3) with added 1s
    X = np.hstack((X,np.ones((X.shape[0],1))))

    ## 3: dot product of X with weights0
    d1 = np.dot(X, w0)

    ## 4: apply sigmoid activation function to each value in d1
    s1 = sigmoid(d1)

    ## 5: add bias column to the hidden layer -> this makes it (50,3), incl. bias
    h1 = np.hstack((s1, np.ones((s1.shape[0],1))))

    ## 6: dot pdocut of hidden layer h1 and weights1
    d2 = np.dot(h1, w1)

    ## 7: apply sigmoid function to get the output layer
    o2 = sigmoid(d2)

    return h1, o2
