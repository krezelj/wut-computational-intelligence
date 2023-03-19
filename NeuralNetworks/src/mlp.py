import numpy as np
from sklearn.metrics import mean_squared_error
from src.layer import Layer


class MLP():

    __slots__ = ['layers', 'first_fit']

    def __init__(self, layers):
        self.layers = layers
        self.first_fit = True

    def predict(self, input, remember_data=False):
        output = input
        for layer in self.layers:
            output = layer.forward(output, remember_data=remember_data)

        return output

    def fit(self, X, Y,
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            momentum_decay_rate=0.9,
            squared_gradient_decay_rate=0.999,
            warm_start=True,
            loss_function='mse',
            verbose=0,
            ):

        # initialise variables
        batches_per_episode = int(np.ceil(X.shape[1] / batch_size))
        loss = []

        epoch = 0
        iteration = 0

        # if warmstart False reset weights for each layer??
        if not warm_start or self.first_fit:
            self.first_fit = False
            for layer in self.layers:
                layer.reset_momentum()

        # fit loop
        while epoch < epochs:
            epoch += 1

            # reorder samples for each episode
            random_indices = np.arange(0, X.shape[1])
            np.random.shuffle(random_indices)
            X = X[:, random_indices]
            Y = Y[:, random_indices]

            # fitting
            for batch_idx in range(batches_per_episode):
                iteration += 1

                batch_start_idx = batch_idx * batch_size
                batch_end_idx = (batch_idx + 1) * batch_size

                x = X[:, batch_start_idx:batch_end_idx]
                y = Y[:, batch_start_idx:batch_end_idx]
                y_predicted = self.predict(x, remember_data=True)

                error = y_predicted - y
                for layer in self.layers[::-1]:
                    error *= layer.activation.derivative(layer.last_output)
                    layer.backward(error, momentum_decay_rate,
                                   squared_gradient_decay_rate)
                    error = np.transpose(layer.weights) @ error

                # apply new weights
                for layer in self.layers:
                    layer.update_weights(learning_rate)

            # calculate loss after epoch
            loss.append(mean_squared_error(Y, self.predict(X)))

            if verbose > 0 and epoch % 2000 == 0:
                print(f"epoch: {epoch}/{epochs}\tloss: {loss[-1]}")

        print(f"done! final loss: {loss[-1]}")
        return loss

    def save_model(self, path, model_name=""):
        full_path = path + '/' + model_name

        data = {}
        # TODO Add Enums for weights, biases, activations
        for i, layer in enumerate(self.layers):
            data[f'{i}_0_weights'] = layer.weights
            data[f'{i}_1_biases'] = layer.biases
            data[f'{i}_2_activation_name'] = np.array([layer.activation_name])

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
            if key[2] == '0':  # name encoding meaning weights
                weights[layer_idx] = value
            elif key[2] == '1':  # name encoding meaning biases
                biases[layer_idx] = value
            elif key[2] == '2':  # name encoding meaning activation
                activations[layer_idx] = value[0]

        layers = [Layer(w.shape[1], w.shape[0], weights=w, biases=b, activation_name=a)
                  for w, b, a in zip(weights, biases, activations)]

        return MLP(layers=layers)


def main():
    import pandas as pd
    df_training = pd.read_csv(
        "NeuralNetworks/data/mio1/regression/square-simple-training.csv", index_col=0)
    df_test = pd.read_csv(
        "NeuralNetworks/data/mio1/regression/square-simple-test.csv", index_col=0)

    x_train = df_training['x'].values.reshape(1, 100)
    y_train = df_training['y'].values.reshape(1, 100)

    model = MLP(layers=[
        Layer(1, 6),
        Layer(6, 1, activation="linear")
    ])

    model.fit(x_train, y_train, learning_rate=0.001,
              epochs=1e1, verbose=1, batch_size=25)
    model.save_model(".", "test")
    MLP.load_model(".", "test")


if __name__ == '__main__':
    main()
