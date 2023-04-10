import numpy as np
from .step import Step


class Dropout(Step):

    __slots__ = ['dropout_p', 'mask']

    def __init__(self, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p

    def forward(self, inputs):
        if self.training:
            self.mask = np.random.binomial(
                1, 1 - self.dropout_p, size=inputs.shape) / (1 - self.dropout_p)
            return self.mask * inputs
        else:
            return inputs

    def backward(self, gradient):
        if self.training:
            return self.mask * gradient
        else:
            return gradient
