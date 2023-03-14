import numpy as np
from src.activations import get_activation_by_name


class Layer():

    __slots__ = ['weights', 'biases', 'activation',
                 'input_dim', 'output_dim', 'last_input', 'last_output', 'delta_weights', 'delta_biases']

    def __init__(self, input_dim, output_dim, weights=None, biases=None, activation_name="sigmoid"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = get_activation_by_name(activation_name)

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

    def forward(self, input, remember_data=False):
        output = self.activation(self.weights @ input + self.biases)
        if remember_data:
            self.last_input = input
            self.last_output = output
        return output

    def backward(self, error, learning_rate=1e-3):
        batch_size = error.shape[1]
        self.delta_weights = -learning_rate * \
            error @ np.transpose(self.last_input) / batch_size
        self.delta_biases = - \
            np.mean(learning_rate * error, axis=1, keepdims=True)
        new_error = np.transpose(self.weights) @ error
        return new_error

    def apply_new_weights(self):
        self.weights += self.delta_weights
        self.biases += self.delta_biases

    def __init_weights(self):
        self.weights = np.random.uniform(-1, 1,
                                         (self.output_dim, self.input_dim))

    def __init_biases(self):
        self.biases = np.random.uniform(-1, 1, (self.output_dim, 1))
