import numpy as np

'''
Loss function and its derivative

Make sure to give the y_true and y_pred values in the fit() method of network.py in the correct order as the function below.
I gave the wrong order and the errors started to increase instead of decreasing.
'''

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size