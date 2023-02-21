import numpy as np
from scipy.special import expit, softmax


class MLP():

    __slots__ = ['layers']

    def __init__(self, layers):
        self.layers = layers

    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)

        return output


class Layer():

    __slots__ = ['weights', 'biases', 'activation', 'input_dim', 'output_dim']

    def __init__(self, input_dim, output_dim, weights=None, biases=None, activation="tanh"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        if weights is None:
            self.__init_weights()
        else:
            assert(weights.shape == (output_dim, input_dim))
            self.weights = weights

        if biases is None:
            self.__init_biases()
        else:
            assert(biases.shape == (output_dim,))
            self.biases = biases

    def forward(self, input):
        output = self.weights @ input + self.biases
        return self.__activate(output)

    def __activate(self, values):
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
        self.weights = np.random.random((self.output_dim, self.input_dim))

    def __init_biases(self):
        self.biases = np.random.random(self.output_dim)


def main():
    model = MLP(layers=[
        Layer(2, 1, activation="linear", weights=np.array(
            [[1, -1]]), biases=np.array([2]))
    ])
    values = np.array([1, 2])
    print(model.predict(values))


if __name__ == '__main__':
    main()
