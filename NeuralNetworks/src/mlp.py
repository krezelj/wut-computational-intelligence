import numpy as np
import networkx as nx
from scipy.special import expit, softmax
from sklearn.metrics import mean_squared_error


class MLP():

    __slots__ = ['layers']

    def __init__(self, layers):
        self.layers = layers

    def predict(self, input, remember_data=False):
        output = input
        for layer in self.layers:
            output = layer.forward(output, remember_data=remember_data)

        return output

    def fit(self, X, Y, learning_rate=1e-3, epochs=1, loss_function='mse', batch_size=1, verbose=0):
        epoch = 0
        batches_per_episode = int(np.ceil(X.shape[1] / batch_size))

        loss = []
        while epoch < epochs:
            if verbose > 0 and epoch % 2000 == 0:
                print(f"epoch: {epoch}/{epochs}")
            epoch += 1

            # reorder samples for each episode
            random_indices = np.arange(0, X.shape[1])
            np.random.shuffle(random_indices)
            X = X[:, random_indices]
            Y = Y[:, random_indices]

            # fitting
            for batch_idx in range(batches_per_episode):
                batch_start_idx = batch_idx * batch_size
                batch_end_idx = (batch_idx + 1) * batch_size

                x = X[:, batch_start_idx:batch_end_idx]
                y = Y[:, batch_start_idx:batch_end_idx]
                y_predicted = self.predict(x, remember_data=True)

                # TODO To clean up
                error = y_predicted - y
                for layer in self.layers[::-1]:
                    error *= layer.activation_derivative(layer.last_output)
                    error = layer.backward(error, learning_rate)

                # apply new weights
                for layer in self.layers:
                    layer.apply_new_weights()

            # calculate loss after epoch
            loss.append(mean_squared_error(Y, self.predict(X)))

        print("done!")
        return loss
    

    def save_model(self, path, model_name=""):
        full_path = path + '/' + model_name

        data = {}
        # TODO Add Enums for weights, biases, activations
        for i, layer in enumerate(self.layers):
            data[f'{i}_0_weights'] = layer.weights
            data[f'{i}_1_biases'] = layer.biases
            data[f'{i}_2_activation'] = np.array([layer.activation])

        np.savez(full_path, **data)

    @classmethod
    def load_model(cls, path, model_name=""):
        full_path = path + '/' + model_name + '.npz'
        data = np.load(full_path)

        # TODO Parametrise the number of attributes remembered per layer
        weights = [None] * (len(data) // 3)
        biases = [None] * (len(data) // 3)
        activations = [None] * (len(data) // 3)
        for key, value in data.items():
            layer_idx = ord(key[0]) - 48
            if key[2] == '0': # name encoding meaning weights
                weights[layer_idx] = value
            elif key[2] == '1': # name encoding meaning biases
                biases[layer_idx] = value
            elif key[2] == '2': # name encoding meaning activation
                activations[layer_idx] = value[0]

        
        layers = [Layer(w.shape[1], w.shape[0], weights=w, biases=b, activation=a) 
                  for w, b, a in zip(weights, biases, activations)]

        return MLP(layers=layers)



class Layer():

    __slots__ = ['weights', 'biases', 'activation',
                 'input_dim', 'output_dim', 'last_input', 'last_output', 'delta_weights', 'delta_biases']

    def __init__(self, input_dim, output_dim, weights=None, biases=None, activation="sigmoid"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

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
        output = self.__activate(self.weights @ input + self.biases)
        if remember_data:
            self.last_input = input
            self.last_output = output
        return output

    def backward(self, error, learning_rate=1e-3):
        batch_size = error.shape[1]
        self.delta_weights = -learning_rate * error @ np.transpose(self.last_input) / batch_size
        self.delta_biases = -np.mean(learning_rate * error, axis=1, keepdims=True)
        new_error = np.transpose(self.weights) @ error
        return new_error

    def activation_derivative(self, values):
        # TODO parametrise epsilon
        eps = 1e-3
        if self.activation == "tanh":
            return 1 - values**2
        elif self.activation == "sigmoid":
            return values * (1 - values)
        elif self.activation == "relu":
            return (values > 0) * (1 - eps) + eps
        elif self.activation == "linear":
            return np.ones(shape=values.shape)
        elif self.activation == "softmax":
            # TODO implement softmax derivatives
            raise NotImplementedError()

    def apply_new_weights(self):
        self.weights += self.delta_weights
        self.biases += self.delta_biases

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
        self.weights = np.random.uniform(-1, 1, (self.output_dim, self.input_dim))

    def __init_biases(self):
        self.biases = np.random.uniform(-1, 1, (self.output_dim, 1))



def main():
    import pandas as pd
    df_training = pd.read_csv("NeuralNetworks/data/mio1/regression/square-simple-training.csv", index_col=0)
    df_test = pd.read_csv("NeuralNetworks/data/mio1/regression/square-simple-test.csv", index_col=0)

    x_train = df_training['x'].values.reshape(1, 100)
    y_train = df_training['y'].values.reshape(1, 100)

    model = MLP(layers=[
        Layer(1, 6),
        Layer(6, 1, activation="linear")
    ])

    model.fit(x_train, y_train, learning_rate=0.001, epochs=1e1, verbose=1, batch_size=25)
    model.save_model(".", "test")
    MLP.load_model(".", "test")


if __name__ == '__main__':
    main()
