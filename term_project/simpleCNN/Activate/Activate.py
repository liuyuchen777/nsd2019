import numpy as np
def ReLU(x):
    return np.where(x > 0, x, 0)
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
def tanh(x):
    return np.tanh(x)
def Leaky_ReLU(x,lamda = 0.01):
    return np.where(x > 0, x, lamda * x)
def Softplus(x):
    return np.log(1.0 + np.exp(-x))