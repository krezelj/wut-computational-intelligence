import numpy as np
from src.activations import get_activation_by_name


class Layer():

    __slots__ = ['weights', 'biases',
                 'input_dim', 'output_dim',
                 'last_input', 'last_output',
                 'momentum_weights', 'momentum_biases',
                 'gradient_weights_squared', 'gradient_biases_squared']

    def __init__(self, input_dim, output_dim, weights=None, biases=None):
        self.input_dim = input_dim
        self.output_dim = output_dim

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

    def forward(self, input):
        self.last_input = input
        return self.weights @ input + self.biases

    def backward(self, gradient, momentum_decay_rate=0.9, squared_gradient_decay_rate=0.999):
        batch_size = gradient.shape[1]

        # calculate batch gradient
        gradient_weights = gradient @ np.transpose(
            self.last_input) / batch_size
        gradient_biases = np.mean(gradient, axis=1, keepdims=True)

        # update momentum
        self.momentum_weights = momentum_decay_rate * self.momentum_weights + \
            (1 - momentum_decay_rate) * gradient_weights
        self.momentum_biases = momentum_decay_rate * self.momentum_biases + \
            (1 - momentum_decay_rate) * gradient_biases

        # update second moment estimate
        self.gradient_weights_squared = squared_gradient_decay_rate * self.gradient_weights_squared + \
            (1 - squared_gradient_decay_rate) * \
            np.square(gradient_weights)
        self.gradient_biases_squared = squared_gradient_decay_rate * self.gradient_biases_squared + \
            (1 - squared_gradient_decay_rate) * \
            np.square(gradient_biases)

        return np.transpose(self.weights) @ gradient

    def update_weights(self, iteration,
                       learning_rate=1e-3,
                       momentum_decay_rate=0.9,
                       squared_gradient_decay_rate=0.999,
                       eps=1e-7):

        # correct bias
        momentum_weights_corrected = self.momentum_weights / \
            (1 - momentum_decay_rate ** iteration)
        momentum_biases_corrected = self.momentum_biases / \
            (1 - momentum_decay_rate ** iteration)
        gradient_weights_squared_corrected = self.gradient_weights_squared / \
            (1 - squared_gradient_decay_rate ** iteration)
        gradient_biases_squared_corrected = self.gradient_biases_squared / \
            (1 - squared_gradient_decay_rate ** iteration)

        self.weights += -learning_rate * momentum_weights_corrected / \
            np.sqrt(gradient_weights_squared_corrected + eps)
        self.biases += -learning_rate * momentum_biases_corrected / \
            np.sqrt(gradient_biases_squared_corrected + eps)

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
