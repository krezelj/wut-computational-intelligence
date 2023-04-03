import numpy as np
from src.layer import Layer
from src.losses import get_loss_function_by_name


class MLP():

    __slots__ = ['steps', 'layers', 'optimiser', 'first_fit']

    def __init__(self, steps, optimiser):
        self.steps = steps
        self.first_fit = True

        self.layers = []
        for step in self.steps:
            if type(step) is Layer:
                self.layers.append(step)

        self.optimiser = optimiser
        self.optimiser.set_layers(self.layers)

    def predict(self, input):
        output = input
        for step in self.steps:
            output = step.forward(output)
        return output

    def fit(self, X, Y,
            epochs=1,
            batch_size=1,
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

        if not warm_start or self.first_fit:
            self.first_fit = False
            self.optimiser.reset()

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

                self.optimiser.step()

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
