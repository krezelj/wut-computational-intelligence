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
            for j in range(neurons_in_layer[i]):
                vertex_idx = j + vertex_idx_offset
                G.nodes[vertex_idx]['pos'] = (i, j + 1 - offset)
            vertex_idx_offset += neurons_in_layer[i]
            bias_vertex_idx += 1

        return G

    def plot(self):
        G = self.__get_graph_representation()
        pos = nx.get_node_attributes(G, 'pos')
        weights = nx.get_edge_attributes(G, 'weight')
        weight_values = np.log10(np.abs(list(weights.values())) + 1)
        edge_color_map = ['blue' if value > 0 else 'red'
                          for value in list(weights.values())]

        nodelist = G.nodes()

        nx.draw_networkx_nodes(G, pos,
                               nodelist=nodelist,
                               node_color='black',)
        nx.draw_networkx_edges(G, pos,
                               edgelist=weights.keys(),
                               width=weight_values,
                               edge_color=edge_color_map)


class Layer():

    __slots__ = ['weights', 'biases', 'activation', 'input_dim', 'output_dim']

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
    W1 = np.array([
        [-3.2], [-4.5], [3.2], [4.5], [0]
    ])
    B1 = np.array([
        -3, -7.62, -3, -7.62, 0
    ]).reshape(5, -1)
    W2 = np.array([
        [170, 242, 170, 242, 0]
    ])
    B2 = np.array([
        -145
    ]).reshape(1, -1)

    model = MLP(layers=[
        Layer(1, 5, W1, B1),
        Layer(5, 1, W2, B2, activation="linear")
    ])
    model.plot()


if __name__ == '__main__':
    main()
