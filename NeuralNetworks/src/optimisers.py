import numpy as np


class Optimiser():

    __slots__ = ['learning_rate', 'layers']

    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate

    def set_layers(self, layers):
        self.layers = layers
        self.reset()

    def reset(self):
        raise NotImplementedError


class SGD(Optimiser):

    __slots__ = ['momentum', 'momentum_weights', 'momentum_biases']

    def __init__(self, learning_rate=1e-3, momentum=0) -> None:
        super().__init__(learning_rate)
        self.momentum = momentum

    def reset(self):
        if self.layers is None:
            raise Exception()

        n_layers = len(self.layers)
        self.momentum_weights = [0] * n_layers
        self.momentum_biases = [0] * n_layers

    def step(self):
        if self.layers is None:
            raise Exception()

        for i, layer in enumerate(self.layers):
            self.momentum_weights[i] = self.momentum * \
                self.momentum_weights[i] + layer.gradient_weights
            self.momentum_biases[i] = self.momentum * \
                self.momentum_biases[i] + layer.gradient_biases

            layer.weights += -self.learning_rate * self.momentum_weights[i]
            layer.biases += -self.learning_rate * self.momentum_biases[i]


class RMSprop(Optimiser):

    __slots__ = ['decay',
                 'momentum_weights_squared',
                 'momentum_biases_squared']

    def __init__(self, learning_rate=1e-3, decay=0.99) -> None:
        super().__init__(learning_rate)
        self.decay = decay

    def reset(self):
        if self.layers is None:
            raise Exception()

        n_layers = len(self.layers)
        self.momentum_weights_squared = [0] * n_layers
        self.momentum_biases_squared = [0] * n_layers

    def step(self):
        if self.layers is None:
            raise Exception()

        for i, layer in enumerate(self.layers):
            self.momentum_weights_squared[i] = self.decay * self.momentum_weights_squared[i] + \
                (1 - self.decay) * np.square(layer.gradient_weights)
            self.momentum_biases_squared[i] = self.decay * self.momentum_biases_squared[i] + \
                (1 - self.decay) * np.square(layer.gradient_biases)

            layer.weights += -self.learning_rate * \
                (self.gradient_weights[i] /
                 np.sqrt(self.momentum_weights_squared[i]))
            layer.biases += -self.learning_rate * \
                (self.gradient_biases[i] /
                 np.sqrt(self.momentum_biases_squared[i]))


class Adam(Optimiser):

    __slots__ = ['beta_1', 'beta_2', 'eps', 'iteration',
                 'momentum_weights', 'momentum_biases',
                 'gradient_weights_squared', 'gradient_biases_squared']

    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-8) -> None:
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.iteration = 0

    def reset(self):
        if self.layers is None:
            raise Exception()

        n_layers = len(self.layers)
        self.momentum_weights = [0] * n_layers
        self.momentum_biases = [0] * n_layers
        self.gradient_weights_squared = [0] * n_layers
        self.gradient_biases_squared = [0] * n_layers

    def step(self):
        if self.layers is None:
            raise Exception()

        self.iteration += 1
        for i, layer in enumerate(self.layers):
            # update momentum
            self.momentum_weights[i] = self.beta_1 * self.momentum_weights[i] + \
                (1 - self.beta_1) * layer.gradient_weights
            self.momentum_biases[i] = self.beta_1 * self.momentum_biases[i] + \
                (1 - self.beta_1) * layer.gradient_biases

            # update second moment estimate
            self.gradient_weights_squared[i] = self.beta_2 * self.gradient_weights_squared[i] + \
                (1 - self.beta_2) * np.square(layer.gradient_weights)
            self.gradient_biases_squared[i] = self.beta_2 * self.gradient_biases_squared[i] + \
                (1 - self.beta_2) * np.square(layer.gradient_biases)

            # correct bias
            momentum_weights_corrected = self.momentum_weights[i] / \
                (1 - self.beta_1 ** self.iteration)
            momentum_biases_corrected = self.momentum_biases[i] / \
                (1 - self.beta_1 ** self.iteration)
            gradient_weights_squared_corrected = self.gradient_weights_squared[i] / \
                (1 - self.beta_2 ** self.iteration)
            gradient_biases_squared_corrected = self.gradient_biases_squared[i] / \
                (1 - self.beta_2 ** self.iteration)

            layer.weights += -self.learning_rate * momentum_weights_corrected / \
                np.sqrt(gradient_weights_squared_corrected + self.eps)
            layer.biases += -self.learning_rate * momentum_biases_corrected / \
                np.sqrt(gradient_biases_squared_corrected + self.eps)
