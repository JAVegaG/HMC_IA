import copy
import numpy as np
from sklearn.metrics import accuracy_score

class logitRegression():
  def __init__(self):
    self.losses = []
    self.train_accuracies = []

  def fit(self, x, y, epochs, learningRate=0.1, optimizationAlgorithm='SGD',
          momentum=0.9):
    """
    Given X and Y, where X is the input matrix of the net, and Y is the actual class expected
    In this model, the input matrix X has a shape(x,w), where x is the number of instances
    per each one of the attributes w.

    Weights and Bias are initialized as zeros
    """
    x = self._transform_x(x)
    y = self._transform_y(y)
  
    self.weights = np.zeros(x.shape[1])
    self.bias = 0

    self.change_w = np.zeros(x.shape[1])
    self.change_b = 0

    self.cache_w = np.zeros(x.shape[1])
    self.cache_b = 0

    for i in range(epochs):
        z = np.matmul(self.weights, x.transpose()) + self.bias
        pred = self._sigmoid(z)
        loss = self.compute_loss(y, pred)
        self.compute_optimization(x, y, pred, optimizationAlgorithm, learningRate, momentum)

        pred_to_class = [1 if p > 0.5 else 0 for p in pred]
        self.train_accuracies.append(accuracy_score(y, pred_to_class))
        self.losses.append(loss)

  def compute_loss(self, y_true, y_pred):
    # binary cross entropy
    y_zero_loss = y_true * np.log(y_pred + 1e-9)
    y_one_loss = (1-y_true) * np.log(1 - y_pred + 1e-9)
    return -np.mean(y_zero_loss + y_one_loss)

  def compute_optimization(self, x, y_true, y_pred, optimizationAlgorithm,
                           learningRate, momentum):
    # derivative of binary cross entropy
  
    error_y =  y_pred - y_true
    gradient_b = np.mean(error_y)
    gradients_w = np.matmul(x.transpose(), error_y)
    gradients_w = np.array([np.mean(grad) for grad in gradients_w])

    SGD_w = learningRate * gradients_w
    SGD_b = learningRate * gradient_b
    
    if optimizationAlgorithm == 'SGD':

      self.weights -= SGD_w
      self.bias -= SGD_b

    elif optimizationAlgorithm == 'SGDM':

      self.change_w = momentum * self.change_w - SGD_w
      self.change_b = momentum * self.change_b - SGD_b

      self.weights += self.change_w
      self.bias += self.change_b

    elif optimizationAlgorithm == 'AdaGrad':

      self.cache_w += gradients_w**2
      self.cache_b += gradient_b**2

      self.weights -= -learningRate * gradients_w / (np.sqrt(self.cache_w) + 1e-6)
      self.bias -= -learningRate * gradients_w / (np.sqrt(self.cache_b) + 1e-6)
      
    else:
      self.weights -= SGD_w
      self.bias -= SGD_b

  def predict(self, x):
    z = np.matmul(x, self.weights.transpose()) + self.bias
    probabilities = self._sigmoid(z)
    return [1 if p > 0.5 else 0 for p in probabilities]

  def _sigmoid(self, x):
    return np.array([self._sigmoid_function(value) for value in x])

  def _sigmoid_function(self, x):
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)

  def _transform_x(self, x):
    """
    This function returns only the values contained by the pandas dataframe x
    as an array
    """
    x = copy.deepcopy(x)
    return x.values

  def _transform_y(self, y):
    """
    This function returns only the values contained by the pandas series y
    as a column vector
    """
    y = copy.deepcopy(y)
    return y.values.reshape(y.shape[0], 1)