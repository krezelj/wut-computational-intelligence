import numpy as np
from scipy.special import expit, softmax


class MLP():

    __slots__ = ['layers']

    def __init__(self, steps):
        pass

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)

        return output


class Layer():

    __slots__ = ['weights', 'biases', 'activation', 'input_dim', 'output_dim']

    def __init__(self, input_dim, output_dim, weights=None, biases=None, activation="tanh"):
        self.input_dim = input_dim
        self.output_dim = output_dim

        if weights is None:
            self.__init_weights()
        else:
            assert(weights.shape == (input_dim, output_dim))
            self.weights = weights

        if biases is None:
            assert(biases.shape == (output_dim))
            self.__init_biases()
        else:
            self.biases = biases

    def forward(self, input):
        output = self.weights @ input + self.biases
        return self.__activation(output)

    def __activation(self, values):
        if self.activation == "tanh":
            return np.tanh(values)
        elif self.activation == "sigmoid":
            return expit(values)
        elif self.activation == "relu":
            return np.maximum(0, values)
        elif self.activation == "linear":
            return values
        elif self.activation == "softmax":
            return softmax(values)
        return values

    def __init_weights(self):
        self.weights = np.random.random((self.input_dim, self.output_dim))

    def __init_biases(self):
        self.weights = np.random.random(self.output_dim)
