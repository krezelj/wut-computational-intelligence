import numpy as np
import networkx as nx
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

    def fit(self, X, Y, learning_rate=1e-3, loss_function='mse', batch_size=1):
        iterations = 0
        while iterations < 1000:
            iterations += 1
            for i in range(X.shape[1]):
                x = X[:, i].reshape(-1, 1)
                y = Y[:, i].reshape(-1, 1)
                y_predicted = self.predict(x)

                # propagate backwards
                error = y_predicted - y
                for layer in self.layers[::-1]:
                    error *= layer.get_derivative_at_last_output()
                    error = layer.backward(error, learning_rate)

                # apply new weights
                for layer in self.layers:
                    layer.apply_new_weights()

    def __get_graph_representation(self) -> nx.DiGraph:
        G = nx.DiGraph()

        edges = []
        neurons_in_layer = [layer.input_dim for layer in self.layers]
        neurons_in_layer.append(self.layers[-1].output_dim)
        total_neuron_count = sum(neurons_in_layer)
        vertex_idx_offset = 0

        # add edges in all but last layer
        for layer in self.layers:
            for i in range(layer.input_dim):
                input_vertex_idx = i + vertex_idx_offset
                for j in range(layer.output_dim):
                    output_vertex_idx = j + vertex_idx_offset + layer.input_dim
                    edges.append((input_vertex_idx,
                                  output_vertex_idx, layer.weights[j, i]))
            vertex_idx_offset += layer.input_dim

        # add bias edges
        vertex_idx_offset = 0
        bias_vertex_idx = total_neuron_count
        for layer in self.layers:
            vertex_idx_offset += layer.input_dim
            for i in range(layer.output_dim):
                output_vertex_idx = i + vertex_idx_offset
                edges.append((bias_vertex_idx,
                              output_vertex_idx, layer.biases[i, 0]))
            bias_vertex_idx += 1

        G.add_weighted_edges_from(edges)

        # position neurons
        vertex_idx_offset = 0
        bias_vertex_idx = total_neuron_count
        for i in range(len(neurons_in_layer)):
            offset = (neurons_in_layer[i] + 1) / \
                2 - 0.5  # +1 to accommodate bias
            if i < len(neurons_in_layer) - 1:   # no bias in the output layer
                G.nodes[bias_vertex_idx]['pos'] = (i, 0 - offset)
                G.nodes[bias_vertex_idx]['is_bias'] = True
            for j in range(neurons_in_layer[i]):
                vertex_idx = j + vertex_idx_offset
                G.nodes[vertex_idx]['pos'] = (i, j + 1 - offset)
                G.nodes[vertex_idx]['is_bias'] = False
            vertex_idx_offset += neurons_in_layer[i]
            bias_vertex_idx += 1

        return G

    def plot(self, log_weights=False):
        G = self.__get_graph_representation()
        pos = nx.get_node_attributes(G, 'pos')
        is_bias_list = list(nx.get_node_attributes(G, 'is_bias').values())
        node_color_map = [
            '#a0a0a0' if is_bias else '#353535' for is_bias in is_bias_list]

        weights = nx.get_edge_attributes(G, 'weight')

        weight_values = np.abs(list(weights.values()))
        if log_weights:
            weight_values = np.log10(weight_values + 1)
        edge_color_map = ['#284db5' if value > 0 else '#b52828'
                          for value in list(weights.values())]

        nodelist = G.nodes()

        nx.draw_networkx_nodes(G, pos,
                               nodelist=nodelist,
                               node_color=node_color_map)
        nx.draw_networkx_edges(G, pos,
                               edgelist=weights.keys(),
                               width=weight_values,
                               edge_color=edge_color_map)


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
            assert(weights.shape == (output_dim, input_dim))
            self.weights = weights

        if biases is None:
            self.__init_biases()
        else:
            assert(biases.shape == (output_dim, 1))
            self.biases = biases

    def forward(self, input):
        self.last_input = input
        output = self.__activate(self.weights @ input + self.biases)
        self.last_output = output
        return output

    def backward(self, error, learning_rate=1e-3):
        self.delta_weights = -learning_rate * \
            error * np.transpose(self.last_input)
        self.delta_biases = -learning_rate * error
        return np.transpose(self.weights) * error

    def get_derivative_at_last_output(self):
        # TODO rename this function to something better
        # TODO parametrise epsilon
        eps = 1e-3
        if self.activation == "tanh":
            return 1 - self.last_output**2
        elif self.activation == "sigmoid":
            return self.last_output * (1 - self.last_output)
        elif self.activation == "relu":
            return (self.last_output > 0) * (1 - eps) + eps
        elif self.activation == "linear":
            return np.ones(shape=self.last_output.shape)
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
        self.weights = np.random.random((self.output_dim, self.input_dim))

    def __init_biases(self):
        self.biases = np.random.random((self.output_dim, 1))


def main():
    model = MLP(layers=[
        Layer(2, 2, activation="linear")
    ])

    x = np.linspace(-1, 1, 100).reshape(1, 100)
    x = np.concatenate([x, x])
    y = 2 * x[0, :] + 1 * x[0, :].reshape(1, -1)

    model.fit(x, y)


if __name__ == '__main__':
    main()
