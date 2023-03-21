import numpy as np
from scipy.special import expit, softmax


def get_activation_by_name(activation):
    if activation == "sigmoid":
        return Sigmoid()
    elif activation == "tanh":
        return Tanh()
    elif activation == "relu":
        return ReLU()
    elif activation == "linear":
        return Linear()
    elif activation == "softmax":
        return Softmax()
    return activation


class Linear():

    def __call__(self, values):
        return values

    def derivative(self, values, activated=True):
        # if activated set to True it's assumed that values are already an output of the sigmoid function
        if not activated:
            values = self(values)
        return np.ones(shape=values.shape)


class Sigmoid():

    def __call__(self, values):
        return expit(values)

    def derivative(self, values, activated=True):
        # if activated set to True it's assumed that values are already an output of the sigmoid function
        if not activated:
            values = self(values)
        return values * (1 - values)


class Tanh():

    def __call__(self, values):
        return np.tanh(values)

    def derivative(self, values, activated=True):
        # if activated set to True it's assumed that values are already an output of the tanh function
        if not activated:
            values = self(values)
        return 1 - values**2


class ReLU():

    __slots__ = ['epsilon']

    def __init__(self, epsilon=1e-3):
        self.epsilon = epsilon

    def __call__(self, values):
        return np.maximum(0, values)

    def derivative(self, values, activated=True):
        # if activated set to True it's assumed that values are already an output of the relu function
        if not activated:
            values = self(values)
        return (values > 0) * (1 - self.eps) + self.eps


class Softmax():

    def __call__(self, values):
        return softmax(values, axis=0)

    def derivative(self, values, activated=True):
        # if activated set to True it's assumed that values are already an output of the softmax function
        if not activated:
            values = self(values)
        raise NotImplementedError


def main():
    pass


if __name__ == "__main__":
    main()
