'''
Activation functions.

Note: ReLU is a good alternative to tanh.

def relu(x):
    return np.maximum(x)

def relu_prime(x):
    return np.where(x>0,1,0)

'''

import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2