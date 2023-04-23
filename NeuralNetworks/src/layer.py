import numpy as np
from .step import Step
from .regularisers import *


class Layer(Step):

    __slots__ = ['weights', 'biases', 'regulariser',
                 'input_dim', 'output_dim',
                 'last_input', 'last_output',
                 'gradient_weights', 'gradient_biases']

    def __init__(self, input_dim, output_dim, weights=None, biases=None, regulariser=None):
        self.input_dim = input_dim
        self.output_dim = output_dim

        if regulariser == 'l1' or type(regulariser) is L1:
            self.regulariser = L1()
        elif regulariser == 'l2' or type(regulariser) is L2:
            self.regulariser = L2()
        else:
            self.regulariser = NoRegulariser()

        if weights is None:
            self.__init_weights()
        else:
            assert (weights.shape == (output_dim, input_dim))
            self.weights = weights

        if biases is None:
            self.__init_biases()
        else:
            assert (biases.shape == (output_dim, 1))
            self.biases = biases

    def forward(self, inputs):
        self.last_input = inputs
        return self.weights @ inputs + self.biases

    def backward(self, gradient):
        batch_size = gradient.shape[1]

        self.gradient_weights = gradient @ np.transpose(
            self.last_input) / batch_size
        self.gradient_biases = np.mean(gradient, axis=1, keepdims=True)

        # regularisation
        self.gradient_weights += self.regulariser.compute_gradients(
            self.gradient_weights)
        self.gradient_biases += self.regulariser.compute_gradients(
            self.gradient_biases)

        return np.transpose(self.weights) @ gradient

    def __init_weights(self):
        self.weights = np.random.uniform(-1, 1,
                                         (self.output_dim, self.input_dim))

    def __init_biases(self):
        self.biases = np.random.uniform(-1, 1, (self.output_dim, 1))
