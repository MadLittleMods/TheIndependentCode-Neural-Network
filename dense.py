import numpy as np
from layer import Layer
import logging

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        logging.error("\tweights_gradient %s", weights_gradient)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        logging.error("\tbias: cost gradient %s", output_gradient)
        self.bias -= learning_rate * output_gradient
        logging.error("\tinput_gradient %s", input_gradient)
        return input_gradient
