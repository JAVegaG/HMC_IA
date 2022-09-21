from re import X
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_data(x, y):
    plt.xlabel('house size')
    plt.ylabel('price')
    plt.plot(x[:,0], y, 'bo')
    plt.show()

def normalize(data):
    mu = []
    std = []
    aux_data = data.copy()
    
    for i in range(0,data.shape[1]-1):
        aux_data[:,i] = ((aux_data[:,i] - np.mean(aux_data[:,i]))/np.std(aux_data[:, i]))
        mu.append(np.mean(aux_data[:,i]))
        std.append(np.std(aux_data[:, i]))
    
    return aux_data[:,:2], aux_data[:,2], mu, std

def h(x,theta):
    return np.matmul(x, theta)

def cost_function(x, y, theta):
    return ((h(x, theta)-y).T@(h(x, theta)-y))/(2*y.shape[0])

def gradient_descent(x, y, theta, learning_rate=0.1, num_epochs=10):
    m = x.shape[0]
    J_all = []
    
    for _ in range(num_epochs):
        h_x = h(x, theta)
        cost_ = (1/m)*(x.T@(h_x - y))
        theta = theta - (learning_rate)*cost_
        J_all.append(cost_function(x, y, theta))
        
    return theta, J_all 

def plot_cost(J_all, num_epochs):
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.plot(num_epochs, J_all, 'm', linewidth = "5")
    plt.show()

def test(theta, x, mu, std):

    y = []

    for k in range(0, len(x)-1):
        x[k, 0] = (x[k, 0] - mu[0])/std[0]
        x[k, 1] = (x[k, 1] - mu[1])/std[1]
    
        y.append(theta[0] + theta[1]*x[k, 0] + theta[2]*x[k, 1])
    
    return y