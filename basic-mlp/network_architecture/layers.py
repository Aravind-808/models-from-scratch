import numpy as np

# Base class for the different layers
class Layer:

    def __init__(self):
        self.input = None
        self.output = None
    
    # forward propagation
    def forwardprop(self, input):
        raise NotImplementedError
    
    # backward propagation
    def backwardprop(self, output_error, learning_rate):
        raise NotImplementedError

# FC Layer (linear)
class FullyConnectedLayer(Layer):
    '''
    (i)   input_size is number of input neurons
    (ii)  output size is number of output neurons
    (iii) initializing weights and bias. subtracting 0.5 from result to ensure better training (by starting with weights and bias close to 0)
    ''' 
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
    
    # each neuron contains the summation of all the weights and input values of neurons before it, plus bias
    def forwardprop(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    '''
    -> given a error derivative dE/dY, we calculate backward propagation and return dE/dX
        (i)   dy_error = output error
        (ii)  dx_error = input error (what we want)
        (iii) dw_error = weights error
    Note: dE/dX for current layer is dE/dY for the previous layer.
    '''
    def backwardprop(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.reshape(-1, 1), output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

# Activaton layer!!

class ActivationLayer(Layer):
    '''
    -> activation function is applied to input during backward propagation in activation layer.
        (i)  activation prime applies the function 1 - tan^2(x)
        (ii) activation applies tan(x)
    '''
    def __init__(self, activation, activation_prime):
        self.activation = activation 
        self.activation_prime = activation_prime

    def forwardprop(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)

        return self.output
    
    def backwardprop(self, op_error, learning_rate):
        return self.activation_prime(self.input)*op_error