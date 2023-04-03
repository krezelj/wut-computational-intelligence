import numpy as np
from src.layer import Layer
from src.losses import *


class MLP():

    __slots__ = ['steps', 'layers', 'first_fit']

    def __init__(self, steps):
        self.steps = steps
        self.first_fit = True
        self.layers = []
        for step in self.steps:
            if type(step) is Layer:
                self.layers.append(step)

    def predict(self, input):
        output = input
        for step in self.steps:
            output = step.forward(output)

        return output

    def fit(self, X, Y,
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            momentum_decay_rate=0.9,
            squared_gradient_decay_rate=0.999,
            warm_start=True,
            loss_function_name='mse',
            verbose=0,
            ):

        # initialise variables
        batches_per_episode = int(np.ceil(X.shape[1] / batch_size))
        loss = []

        epoch = 0
        iteration = 0

        loss_function, d_loss_function = get_loss_function_by_name(
            loss_function_name)

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
                y_predicted = self.predict(x)

                gradient = d_loss_function(y, y_predicted)
                for step in self.steps[::-1]:
                    gradient = step.backward(gradient)

                # apply new weights
                for layer in self.layers:
                    layer.update_weights(
                        iteration, learning_rate, momentum_decay_rate, squared_gradient_decay_rate)

            # calculate loss after epoch
            loss.append(loss_function(Y, self.predict(X)))

            if verbose > 0 and epoch % 500 == 0:
                print(f"epoch: {epoch}/{epochs}\tloss: {loss[-1]}")

        print(f"done! final loss: {loss[-1]}")
        return loss


def main():
    pass


if __name__ == '__main__':
    main()
