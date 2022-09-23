import numpy as np
from sklearn.metrics import accuracy_score

class mlp:
    '''Multilayer perceptron using numpy
    '''
    def __init__(self, layersizes, activations, derivatives, loss, lossderiv):
        """remember config, then initialize array to hold NN parameters without init"""
        # hold NN config
        self.layersizes = tuple(layersizes)
        self.activations = tuple(activations)
        self.derivatives = tuple(derivatives)
        self.loss = loss
        self.lossderiv = lossderiv

        assert len(self.layersizes)-1 == len(self.activations), \
            "number of layers and the number of activation functions does not match"
        assert len(self.activations) == len(self.derivatives), \
            "number of activation functions and number of derivatives does not match"
        assert all(isinstance(n, int) and n >= 1 for n in layersizes), \
            "Only positive integral number of perceptons is allowed in each layer"

        # parameters, each is a 2D numpy array
        L = len(self.layersizes)
        self.z = [None] * L
        self.W = [None] * L
        self.b = [None] * L
        self.a = [None] * L
        self.dz = [None] * L
        self.dW = [None] * L
        self.db = [None] * L
        self.da = [None] * L

        self.delta_W = [None] * L
        self.delta_b = [None] * L

    def initialize(self, seed=42):
        """initialize the value of weight matrices and bias vectors with small random numbers."""
        np.random.seed(seed)
        sigma = 0.1
        for l, (insize, outsize) in enumerate(zip(self.layersizes, self.layersizes[1:]), 1):
            self.W[l] = np.random.randn(insize, outsize) * sigma
            self.b[l] = np.random.randn(1, outsize) * sigma

            self.delta_W[l] = np.random.randn(insize, outsize) * sigma
            self.delta_b[l] = np.random.randn(1, outsize) * sigma

    def forward(self, x):
        """Feed forward using existing `W` and `b`, and overwrite the result variables `a` and `z`

        Args:
            x (numpy.ndarray): Input data to feed forward
        """
        self.a[0] = x
        for l, func in enumerate(self.activations, 1):
            # z = W a + b, with `a` as output from previous layer
            # `W` is of size rxs and `a` the size sxn with n the number of data instances, `z` the size rxn
            # `b` is rx1 and broadcast to each column of `z`
            self.z[l] = (self.a[l-1] @ self.W[l]) + self.b[l]
            # a = g(z), with `a` as output of this layer, of size rxn
            self.a[l] = func(self.z[l])
        return self.a[-1]

    def backward(self, y_true, y_pred):
        """back propagation using NN output y_pred and the reference output y, generates dW, dz, db,
        da
        """
        assert y_true.shape[1] == self.layersizes[-1], "Output size doesn't match network output size"
        assert y_true.shape == y_pred.shape, "Output size doesn't match reference"

        # first `da`, at the output
        self.da[-1] = self.lossderiv(y_true, y_pred)
        
        for l, func in reversed(list(enumerate(self.derivatives, 1))):
            # compute the differentials at this layer
            self.dz[l] = self.da[l] * func(self.z[l])
            self.dW[l] = self.a[l-1].T @ self.dz[l]
            self.db[l] = np.mean(self.dz[l], axis=0, keepdims=True)
            self.da[l-1] = self.dz[l] @ self.W[l].T

            assert self.z[l].shape == self.dz[l].shape
            assert self.W[l].shape == self.dW[l].shape
            assert self.b[l].shape == self.db[l].shape
            assert self.a[l].shape == self.da[l].shape

    def update(self, eta, optimizer, momentum):
        """Updates W and b

        Args:
            eta (float): Learning rate
        """
        for l in range(1, len(self.W)):
            if optimizer == 'GD':
                self.W[l] -= eta * self.dW[l]
                self.b[l] -= eta * self.db[l]
            elif optimizer == 'SGDM':
                self.delta_W[l] = eta * self.dW[l] - momentum * self.delta_W[l]
                self.delta_b[l] = eta * self.db[l] - momentum * self.delta_b[l]

                self.W[l] -= self.delta_W[l]
                self.b[l] -= self.delta_b[l]
            
            elif optimizer == 'AdaGrad':
                self.delta_W[l] += self.dW[l]**2
                self.delta_b[l] += self.db[l]**2

                self.W[l] -= ( eta / ( np.sqrt( self.delta_W[l] + 10e-10 ) ) ) * self.dW
                self.b[l] -= ( eta / ( np.sqrt( self.delta_b[l] + 10e-10 ) ) ) * self.db
            else:
                self.W[l] -= eta * self.dW[l]
                self.b[l] -= eta * self.db[l]
                

    def fit(self, X, y, n_epochs=150, learning_rate=0.005, optimizer='GD',momentum=0.1, verbose=False):
        for n in range(n_epochs):
            y_pred = self.forward(X)
            self.backward(y, y_pred)
            self.update(learning_rate, optimizer, momentum)
            cost = self.loss(y, y_pred)
            if verbose:
                print("Epoch {} - loss value {} accuracy {} Optimizer {}\n".format(n+1,
                cost,
                accuracy_score(y, (y_pred > 0.5).astype(int))
                ,optimizer))
        print("After Training {} epochs\nLoss value {} accuracy {} Optimizer {}\n".format(n+1,
                cost,
                accuracy_score(y, (y_pred > 0.5).astype(int))
                ,optimizer))
