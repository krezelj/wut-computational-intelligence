import numpy as np
from src.activations import get_activation_by_name


class Layer():

    __slots__ = ['weights', 'biases',
                 'activation', 'activation_name',
                 'input_dim', 'output_dim',
                 'last_input', 'last_output',
                 'momentum_weights', 'momentum_biases',
                 'gradient_weights_squared', 'gradient_biases_squared']

    def __init__(self, input_dim, output_dim, weights=None, biases=None, activation_name="sigmoid"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation_name
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

        self.reset_momentum()

    def forward(self, input, remember_data=False):
        output = self.activation(self.weights @ input + self.biases)
        if remember_data:
            self.last_input = input
            self.last_output = output
        return output

    def backward(self, error,
                 momentum_decay_rate,
                 squared_gradient_decay_rate):
        batch_size = error.shape[1]

        # calculate batch gradient
        gradient_weights = error @ np.transpose(self.last_input) / batch_size
        gradient_biases = np.mean(error, axis=1, keepdims=True)

        # update momentum
        self.momentum_weights = momentum_decay_rate * self.momentum_weights + \
            (1 - momentum_decay_rate) * gradient_weights
        self.momentum_biases = momentum_decay_rate * self.momentum_biases + \
            (1 - momentum_decay_rate) * gradient_biases

        # update gradients squared
        self.gradient_weights_squared = squared_gradient_decay_rate * self.gradient_weights_squared + \
            (1 - squared_gradient_decay_rate) * \
            np.square(gradient_weights)
        self.gradient_biases_squared = squared_gradient_decay_rate * self.gradient_biases_squared + \
            (1 - squared_gradient_decay_rate) * \
            np.square(gradient_biases)

    def update_weights(self, learning_rate=1e-3):
        # TODO Parametrise epsilon
        eps = 1e-7
        self.weights += -learning_rate * self.momentum_weights / \
            np.sqrt(self.gradient_weights_squared + eps)
        self.biases += -learning_rate * self.momentum_biases / \
            np.sqrt(self.gradient_biases_squared + eps)

    def reset_momentum(self):
        self.momentum_weights = np.zeros(shape=self.weights.shape)
        self.momentum_biases = np.zeros(shape=self.biases.shape)

        self.gradient_weights_squared = np.zeros(shape=self.weights.shape)
        self.gradient_biases_squared = np.zeros(shape=self.biases.shape)

    def __init_weights(self):
        self.weights = np.random.uniform(-1, 1,
                                         (self.output_dim, self.input_dim))

    def __init_biases(self):
        self.biases = np.random.uniform(-1, 1, (self.output_dim, 1))
