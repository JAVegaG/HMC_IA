import numpy as np

# Sigmoid function and its differentiation
def sigmoid(z):
    z = z.copy()
    return 1/(1+np.exp(-z.clip(-500, 500)))
    
def dsigmoid(z):
    s = sigmoid(z)
    return 2 * s * (1-s)

# ReLU function and its differentiation
def relu(z):
    return np.maximum(0, z)
    
def drelu(z):
    return (z > 0).astype(float)

# Loss function L(y, y_pred) and its differentiation
def cross_entropy(y_true, y_pred):
    """Binary cross entropy function
    """
    epsilon = np.finfo(float).eps
    return -(y_true.T @ np.log(y_pred.clip(epsilon)) + (1-y_true.T) @ np.log((1-y_pred).clip(epsilon))) / y_true.shape[1]

def d_cross_entropy(y_true, y_pred):
    """ dL/dy_pred """
    epsilon = np.finfo(float).eps
    return - np.divide(y_true, y_pred.clip(epsilon)) + np.divide(1-y_true, (1-y_pred).clip(epsilon))