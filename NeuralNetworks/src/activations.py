import numpy as np
from scipy.special import expit, softmax


class Activation():

    __slots__ = ['activation', 'd_activation', 'last_output']

    def __init__(self, activation, d_activation):
        self.activation = activation
        self.activation_prime = d_activation

    def forward(self, input):
        self.last_output = self.activation(input)
        return self.last_output

    def backward(self, gradient):
        return gradient * self.activation_prime(self.last_output, activated=True)


class Linear(Activation):

    def __init__(self):
        super().__init__(self.__call__, self.derivative)

    def __call__(self, values):
        return values

    def derivative(self, values, activated=True):
        # if activated set to True it's assumed that values are already an output of the linear function
        if not activated:
            values = self(values)

        return np.ones(shape=values.shape)


class Sigmoid(Activation):

    def __init__(self):
        super().__init__(self.__call__, self.derivative)

    def __call__(self, values):
        return expit(values)

    def derivative(self, values, activated=True, error=None):
        # if activated set to True it's assumed that values are already an output of the sigmoid function
        if not activated:
            values = self(values)

        return values * (1 - values)


class Tanh(Activation):

    def __init__(self):
        super().__init__(self.__call__, self.derivative)

    def __call__(self, values):
        return np.tanh(values)

    def derivative(self, values, activated=True, error=None):
        # if activated set to True it's assumed that values are already an output of the tanh function
        if not activated:
            values = self(values)

        return 1 - values**2


class ReLU(Activation):

    __slots__ = ['epsilon']

    def __init__(self, epsilon=1e-3):
        self.epsilon = epsilon
        super().__init__(self.__call__, self.derivative)

    def __call__(self, values):
        return np.maximum(0, values)

    def derivative(self, values, activated=True, error=None):
        # if activated set to True it's assumed that values are already an output of the relu function
        if not activated:
            values = self(values)

        return (values > 0) * (1 - self.eps) + self.eps


class Softmax(Activation):

    def __init__(self):
        super().__init__(self.__call__, self.derivative)

    def __call__(self, values):
        return softmax(values, axis=0)

    def derivative(self, values, activated=True, error=None):
        # if activated set to True it's assumed that values are already an output of the softmax function
        if not activated:
            values = self(values)

        # calculate jacobian
        I = np.identity(values.shape[0])
        temp1 = np.einsum('ij,ik->ijk', values, I)
        temp2 = np.einsum('ij,kj->ijk', values, values)
        J = temp1 - temp2
        return J

    def backward(self, gradient):
        return np.einsum('nbk,kb->nb', self.derivative(self.last_output, activated=True), gradient)


def main():
    pass


if __name__ == "__main__":
    main()
