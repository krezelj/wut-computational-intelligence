import numpy as np
from src.layer import Layer
from src.losses import get_loss_function_by_name


class MLP():

    __slots__ = ['steps', 'layers', 'optimiser', 'first_fit', '_training']

    def __init__(self, steps, optimiser):
        self.steps = steps
        self.first_fit = True
        self._training = False
        self.training = False

        self.layers = []
        for step in self.steps:
            if type(step) is Layer:
                self.layers.append(step)

        self.optimiser = optimiser
        self.optimiser.set_layers(self.layers)

    @property
    def training(self):
        return self._training

    @training.setter
    def training(self, value):
        self._training = value
        for step in self.steps:
            step.training = value

    def predict(self, input):
        output = input
        for step in self.steps:
            output = step.forward(output)
        return output

    def fit(self, X_train, Y_train,
            X_validation, Y_validation,
            epochs=1,
            batch_size=1,
            warm_start=True,
            loss_function_name='mse',
            verbose=0,
            ):

        # initialise variables
        batches_per_episode = int(np.ceil(X_train.shape[1] / batch_size))
        validation_loss = []
        train_loss = []

        epoch = 0
        iteration = 0

        loss_function, d_loss_function = get_loss_function_by_name(
            loss_function_name)

        if not warm_start or self.first_fit:
            self.first_fit = False
            self.optimiser.reset()

        # fit loop
        self.training = True
        while epoch < epochs:
            epoch += 1

            # reorder samples for each episode
            random_indices = np.arange(0, X_train.shape[1])
            np.random.shuffle(random_indices)
            X_train = X_train[:, random_indices]
            Y_train = Y_train[:, random_indices]

            # fitting
            for batch_idx in range(batches_per_episode):
                iteration += 1

                batch_start_idx = batch_idx * batch_size
                batch_end_idx = (batch_idx + 1) * batch_size

                x = X_train[:, batch_start_idx:batch_end_idx]
                y = Y_train[:, batch_start_idx:batch_end_idx]
                y_predicted = self.predict(x)

                gradient = d_loss_function(y, y_predicted)
                for step in self.steps[::-1]:
                    gradient = step.backward(gradient)

                self.optimiser.step()

            # calculate loss after epoch
            self.training = False
            validation_loss.append(loss_function(
                Y_validation, self.predict(X_validation)))
            train_loss.append(loss_function(
                Y_train, self.predict(X_train)))
            self.training = True

            # TODO Implement early stopping

            if (verbose > 1 and epoch % 500 == 0) or (verbose > 2):
                print(f"epoch: {epoch}/{epochs}\tloss: {validation_loss[-1]}")

        self.training = False
        if verbose > 0:
            print(f"done! final loss: {validation_loss[-1]}")
        return validation_loss, train_loss


def main():
    pass


if __name__ == '__main__':
    main()
