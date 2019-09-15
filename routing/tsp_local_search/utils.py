import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def compute_tour_edges(dist_mat: np.ndarray, solution: list, depot: int, n_nodes: int):
    tour = []
    n_edges = n_nodes
    node_u = depot

    for edge_idx, edge_idx in enumerate(range(0, n_edges - 1)):
        node_v = solution[edge_idx + 1]
        cost = dist_mat[node_u, node_v]
        edge_info = (edge_idx, cost, (node_u, node_v))
        tour.append(edge_info)
        node_u = node_v

    last_edge = (n_edges, dist_mat[node_v, depot], (node_v, depot))
    tour.append(last_edge)
    return tour

def compute_tour_cost(nodes_lst, distance_matrix):
    cost = 0
    for it, node in enumerate(nodes_lst):
        if it< len(nodes_lst) - 1:
            d = distance_matrix[node, nodes_lst[it + 1]]
            cost += d
        else:
            d = distance_matrix[0, nodes_lst[it]]
            cost += d
    return cost


def plot_route(tour, points_lst):
    edges_lst = []
    for idx_node, node in enumerate(tour):
        if idx_node < len(tour) - 1:
            edges_lst.append((node, tour[idx_node + 1]))

    g = nx.DiGraph(edges_lst)
    plt.figure(figsize=(8, 6))
    node_positions = {idx_node: (val_node.x, val_node.y) for idx_node, val_node in enumerate(points_lst)}
    nx.draw(g, pos=node_positions, node_size=10, node_color='red', with_labels=True)
    nx.draw_networkx_nodes(g, node_positions,
                           nodelist=[0], node_size=50,
                           node_color="b")
    plt.show()