import numpy as np


class Kohonen:
    __slots__ = [
        "weights",
        "self_distances",
        "m",
        "n",
        "k",
        "time_decay",
        "neighbourhood_weight_func",
        "grid_type",
    ]

    def __init__(
        self, m, n, k, time_decay, neighbourhood_weight_func, grid_type="grid"
    ):
        self.m = m
        self.n = n
        self.k = k
        self.time_decay = time_decay
        self.neighbourhood_weight_func = neighbourhood_weight_func
        self.grid_type = grid_type
        self.__init_weights()
        self.__calculate_self_distances()

    def __init_weights(self):
        self.weights = np.random.uniform(low=0, high=1, size=(self.m, self.n, self.k))

    def fit(self, X_train, epochs):
        for epoch in range(epochs):
            # random shuffle of training set
            random_indices = np.arange(0, X_train.shape[0])
            np.random.shuffle(random_indices)
            X_train = X_train[random_indices, :]

            for x in X_train:
                distances_to_x = self.__calculate_distances_squared(x)
                i_bmu, j_bmu = np.unravel_index(
                    distances_to_x.argmin(), distances_to_x.shape
                )

                distances_bmu = self.self_distances[i_bmu, j_bmu]
                self.weights += (
                    self.neighbourhood_weight_func(distances_bmu, epoch).reshape(
                        self.m, self.n, 1
                    )
                    * self.__decay(epoch)
                    * (x - self.weights)
                )

    def match(self, X_train):
        closest_point_idx = np.zeros(shape=self.m * self.n, dtype=np.int32)
        closest_point_distance = np.ones(shape=self.m * self.n) * np.inf
        for i, x in enumerate(X_train):
            distances_to_x = self.__calculate_distances_squared(x).reshape(
                -1,
            )
            idx_to_update = np.where(distances_to_x < closest_point_distance)[0]
            closest_point_idx[idx_to_update] = i
            closest_point_distance[idx_to_update] = distances_to_x[idx_to_update]
        return closest_point_idx

    def __calculate_distances_squared(self, x):
        return np.sum(np.square(self.weights - x, dtype=np.float64), axis=2)

    def __calculate_self_distances(self):
        self.self_distances = np.zeros(shape=(self.m, self.n, self.m, self.n))
        rows = np.arange(self.m)
        columns = np.arange(self.n)
        for i, j, p, q in cartesian_product(rows, columns, rows, columns):
            if self.grid_type == "grid":
                self.self_distances[i, j, p, q] = get_grid_distance(i, j, p, q)
            elif self.grid_type == "hex":
                self.self_distances[i, j, p, q] = get_hex_distance(i, j, p, q)

    def __decay(self, t):
        return np.exp(-(t + 1) * self.time_decay)


def get_grid_distance(i, j, p, q):
    return np.sqrt((i - p) ** 2 + (j - q) ** 2)


def get_hex_distance(i, j, p, q):
    i_ax, j_ax = offset_to_axial(i, j)
    p_ax, q_ax = offset_to_axial(p, q)
    return (abs(i_ax - p_ax) + abs(i_ax + j_ax - p_ax - q_ax) + abs(j_ax - q_ax)) / 2


def offset_to_axial(i, j):
    q = i - (j - (j & 1)) / 2
    r = j
    return q, r


def normalise(A):
    A_sq = np.square(A)
    sums = A_sq.sum(axis=2, keepdims=True)
    return np.sqrt(A_sq / sums, where=sums > 1, out=np.zeros_like(A_sq))


def circle(distances, t=1):
    weights = np.ones(shape=distances.shape)
    weights[np.where(distances > 2)] = 0
    return weights


def gaussian(distances, t=1):
    return np.exp(-np.square(distances / 2))


def mexican_hat(distances, t=1):
    return (2 - 4 * np.square(distances / 6)) * np.exp(-np.square(distances / 6))


def cartesian_product(*arrays):
    meshgrid = np.meshgrid(*arrays, indexing="ij")
    stacked = np.vstack([m.flatten() for m in meshgrid]).T
    return stacked


def main():
    import pandas as pd

    data_hex = pd.read_csv("Kohonen/data/mio/hexagon.csv")
    X_train = data_hex[["x", "y"]].values

    koh = Kohonen(10, 10, 2, 0.9, mexican_hat)
    koh.fit(X_train[:10], 1)


if __name__ == "__main__":
    main()
