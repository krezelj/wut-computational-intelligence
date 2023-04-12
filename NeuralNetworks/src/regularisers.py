import numpy as np


class Regulariser():

    def __init__(self) -> None:
        pass


# Dummy regulariser
class NoRegulariser(Regulariser):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, weights):
        return 0

    def compute_gradients(self, weights):
        return np.zeros(shape=weights.shape)


class L1(Regulariser):

    def __init__(self, l1=0.01) -> None:
        super().__init__()
        self.l1 = l1

    def __call__(self, weights):
        return self.l1 * np.sum(np.abs(weights))

    def compute_gradients(self, weights):
        return self.l1 * np.sign(weights)


class L2(Regulariser):

    def __init__(self, l2=0.01) -> None:
        super().__init__()
        self.l2 = l2

    def __call__(self, weights):
        return self.l2 * np.sum(np.square(weights))

    def compute_gradients(self, weights):
        return self.l2 * weights
