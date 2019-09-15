import pandas as pd
import numpy as np

from routing.tsp_local_search import utils

def swap2opt(distance_matrix, tour, i, j):
    """
    1. take tour[0] to tour[i-1] and add them in order to new_tour
    2. take route[i] to route[j] and add them in reverse order to new_tour
    3. take route[j+1] to end and add them in order to new_tour
    return new_route;
    """
    orig = (i, j)
    if tour.index(i) > tour.index(j):
        i = tour.index(orig[1])
        j = tour.index(orig[0])
    else:
        i = tour.index(i)
        j = tour.index(j)

    new_tour = tour.copy()
    # new_tour[0:i] = tour[0:i].copy()  # [0, i)
    new_tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])  # tour[j-1:i:-1]  #[i,j+1)
    # new_tour[j:] = tour[j:].copy()

    swap_cost = (utils.compute_tour_cost(new_tour[i:j + 2], distance_matrix)) - (
        utils.compute_tour_cost(tour[i:j + 2], distance_matrix))
    # <0 good swap
    return new_tour, swap_cost

def get_m_closest_neighbours(node, tour, distance_matrix, m_neighbours):
    dist_row = distance_matrix[node, :].copy()
    # Subset tour and get nodes ahead the current one
    node_idx = tour.index(node)
    tour_subset = tour[node_idx+1:]
    dist_series = pd.Series([d for idx_d, d in enumerate(dist_row)])
    neighbours = dist_series[tour_subset].sort_values(ascending=True)
    m_neighbours = neighbours[:m_neighbours].index.values
    if len(m_neighbours)==0:
        res = [0]  # Last node
    else:
        res = m_neighbours
    return res

def two_opt_m_neighbours(tour, distance_matrix, m_neighbours, dev: bool = False):
    tour_len = len(tour)
    if m_neighbours > tour_len: m_neighbours = tour_len

    best = tour.copy()
    flg_improved = True
    while flg_improved:
        flg_improved = False
        for i in range(1, tour_len - 2):
            delta_cost_former = 0
            #for j in range(i + 1, i + 1 + m_neighbours):
            closest_neighbours = get_m_closest_neighbours(i, tour, distance_matrix, m_neighbours)
            for j in closest_neighbours:
                if j - i == 1: continue  # changes nothing, skip then
                new_tour, delta_cost = swap2opt(distance_matrix, best, i, j)
                if delta_cost < delta_cost_former:
                    if dev: print(f"improved!!: {delta_cost} {delta_cost_former}, at: {i}, {j}")
                    best = new_tour
                    flg_improved = True
                    delta_cost_former = delta_cost

        tour = best
    return best

def get_most_constrained_points(tour_edges, k_points):
    """
    :param tour_edges: a list of tuples (edge_id, edge_length, (u, v))
    :param k_points: Number of edges to return
    :return: A list of nodes indices that belong to the longest edges
    """
    from operator import itemgetter
    tour_edges_srt = sorted(tour_edges, key=itemgetter(1), reverse=True)
    edges = [x[2] for x in tour_edges_srt]
    nodes = [edge[0] for edge in edges]
    return list(set(nodes[:k_points]))

def get_m_closest_neighbours_mat(distance_matrix, m_neighbours):
    """"
    Build a matrix that for each row, its values represent each node closest neighbours.
    Matrix size is (n, m_neighbours)
    """
    n, _ = np.shape(distance_matrix)
    neig_mat = np.zeros((n, m_neighbours))
    for node in range(0, n):
        row_dist = pd.Series(distance_matrix[node, :]).sort_values(ascending=True)
        row_dist_subset = row_dist[1:m_neighbours+1].copy()
        neig_mat[node, :] = row_dist_subset.index.values
    return neig_mat

def two_opt_m_neighbours_k_most_constrained(tour, tour_edges, distance_matrix, m_neighbours, k_most_constrained,
                                            dev: bool = False):
    """
    :param tour: List of nodes in order of visit
    :param tour_edges: Tuples (idx, cost, (node_u, node_v))
    :param distance_matrix: Symmetric distance matrix
    :param m_neighbours: m_neighbours to perform 2-opt swap for each edge
    :param k_most_constrained: k edges to look for a better connection
    :param dev: If True, detail is printed
    :return: Solution
    """
    tour_len = len(tour)
    if m_neighbours > tour_len: m_neighbours = tour_len
    # Compute a matrix where each row represent each node most closest neighbours
    closest_neig_mat = get_m_closest_neighbours_mat(distance_matrix, m_neighbours)
    # Build a list of nodes that are the longest edges in the input tour
    most_constrained_nodes = get_most_constrained_points(tour_edges, k_most_constrained)
    best = tour.copy()
    # Iterate every node in a long edge
    for it, i in enumerate(most_constrained_nodes):
        delta_cost_former = 0
        closest_neighbours = closest_neig_mat[i]  # Query the neighbours matrix to get node i candidates to swap
        for j in closest_neighbours:
            # if dev: print(i, closest_neighbours)
            if j - i == 1: continue  # changes nothing, skip then
            new_tour, delta_cost = swap2opt(distance_matrix, best, i, j)
            if delta_cost < delta_cost_former:
                if dev: print(f"improved!!: it:{it} {delta_cost} {delta_cost_former}, at: {i}, {j}")
                best = new_tour

                delta_cost_former = delta_cost

    return best