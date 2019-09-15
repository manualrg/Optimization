import math
from collections import namedtuple
from scipy.spatial import distance
import matplotlib.pyplot as plt
import networkx as nx

Point = namedtuple("Point", ['x', 'y'])

def read_input_data(file_path):
    # Modify this code to run your optimization algorithm
    with open(file_path, 'r') as input_data_file:
        input_data = input_data_file.read()
    return input_data

def parse_data(input_data):
    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    return points

def compute_euclidean_distance_matrix(data_points :list, dist_scale_param: int = 1000):
    distance_matrix = distance.cdist(data_points, data_points, 'euclidean')*dist_scale_param
    m = len(data_points)
    dist_dict = {}
    for row in range(0, m):
        dist_dict[row] = {}
        for col in range(0, m):
            dist_dict[row][col] =distance_matrix[row, col]
    return dist_dict, distance_matrix/dist_scale_param

def create_data_model(dist_mat: list):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = dist_mat
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data

def output_solution(manager, routing, assignment, dist_scale_param: int = 1000):
    obj_value = assignment.ObjectiveValue()/dist_scale_param
    index = routing.Start(0)
    route_distance = 0
    node_lst = []
    edges_lst = []
    while not routing.IsEnd(index):
        node_u = manager.IndexToNode(index)
        node_lst.append(node_u)

        previous_index = index
        index = assignment.Value(routing.NextVar(index))
        node_v = manager.IndexToNode(index)
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        edges_lst.append( (node_u, node_v) )
    return obj_value, node_lst, edges_lst


def plot_route(tour, points_lst):
    edges_lst = []
    for idx_node, node in enumerate(tour):
        if idx_node < len(tour) - 1:
            edges_lst.append((node, tour[idx_node + 1]))

    g = nx.DiGraph(edges_lst)
    fig = plt.figure(figsize=(8, 6))
    node_positions = {idx_node: (val_node.x, val_node.y) for idx_node, val_node in enumerate(points_lst)}
    nx.draw(g, pos=node_positions, node_size=10, node_color='red', with_labels=True)
    nx.draw_networkx_nodes(g, node_positions,
                           nodelist=[0], node_size=50,
                           node_color="b")
    plt.show()

def compute_tour_cost(nodes_lst, distance_matrix):
    cost = 0
    for it, node in enumerate(nodes_lst):
        if it < len(nodes_lst) - 1:
            d = distance_matrix[node, nodes_lst[it + 1]]
            cost += d
        else:
            d = distance_matrix[0, nodes_lst[it]]
            cost += d
    return cost