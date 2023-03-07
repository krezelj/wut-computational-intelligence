import networkx as nx
import numpy as np

from src.mlp import MLP, Layer


def get_graph_representation(mlp) -> nx.DiGraph:
    G = nx.DiGraph()

    edges = []
    neurons_in_layer = [layer.input_dim for layer in mlp.layers]
    neurons_in_layer.append(mlp.layers[-1].output_dim)
    total_neuron_count = sum(neurons_in_layer)
    vertex_idx_offset = 0

    # add edges in all but last layer
    for layer in mlp.layers:
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
    for layer in mlp.layers:
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

def plot(mlp, log_weights=False):
    G = get_graph_representation(mlp)
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
    