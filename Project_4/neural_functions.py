from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
import numpy as np

# Sigmoid function and its differentiation
def sigmoid(z):
    z = z.copy()
    return [ ( 1 / (1 + np.exp(-x)) ) if x >= 0 else ( np.exp(x) / (1 + np.exp(x)) ) for x in z]
    
def dsigmoid(z):
    s = sigmoid(z)
    return 2 * s * (1-s)

# ReLU function and its differentiation
def relu(z):
    return np.maximum(0, z)
    
def drelu(z):
    return (z > 0).astype(float)

# Loss function L(y, yhat) and its differentiation
def cross_entropy(y_true, y_pred):
    """Binary cross entropy function
    """
    y_zero_loss = y_true * np.log(y_pred + 1e-9)
    y_one_loss = (1-y_true) * np.log(1 - y_pred + 1e-9)
    return -np.mean(y_zero_loss + y_one_loss)

def d_cross_entropy(y_true, y_pred):
    """ dL/dyhat """
    return - np.divide(y_true, y_pred + 1e-9) + np.divide(1-y_true, (1-y_pred) + 1e-9)