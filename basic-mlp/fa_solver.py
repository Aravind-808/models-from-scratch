import numpy as np
from network_architecture.layers import FullyConnectedLayer, ActivationLayer
from network_architecture.network import Network
from activation_functions.tan_h import tanh, tanh_prime
from error_functions.mean_squared import mse, mse_prime

'''
Usually we test it with the XOR gate table.
But for fun, im gonna test it with the Full adder table lmao.
'''

# training data
x_train = np.array([[0, 0, 0],
                    [0, 0, 1], 
                    [0, 1, 0], 
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1]])
        
y_train = np.array([[0, 0], 
                    [1, 0], 
                    [1, 0], 
                    [0, 1],
                    [1, 0],
                    [0, 1],
                    [0, 1],
                    [1, 1]])

# network
net = Network()

'''
For My network, i am using:
(i)   3 nodes in the input layer
(ii)  4 nodes in the hidden layer
(iii) 2 nodes in the output layer

Using the FClayer and Activation layer stacked together is better because
it introduces non linearity in the neural network which helps in improved
performance.
'''
net.add_layer(FullyConnectedLayer(3, 4))
net.add_layer(ActivationLayer(tanh, tanh_prime))
net.add_layer(FullyConnectedLayer(4, 2))
net.add_layer(ActivationLayer(tanh, tanh_prime))

# train
net.use_loss(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)

res = [arr.flatten().tolist() for arr in out]
print("Output:")
print(res)

res = np.array(res)
res = res.reshape(-2,2)
res = np.round(res,3).astype(float)
print("------------------------------------------------------------------------")
print(f"Output Before rounding off:\n{res}")
print("------------------------------------------------------------------------")
res = np.round(res).astype(int)
print(f"Output after rounding off:\n{res}")
print("------------------------------------------------------------------------")
