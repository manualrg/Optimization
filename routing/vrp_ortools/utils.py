import numpy as np
from scipy.spatial import distance

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.pyplot import cm

import math
from collections import namedtuple

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)


def parse_data(input_data):
    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])

    customers = []
    cust_location = []
    cust_demand = []

    for i in range(1, customer_count + 1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(i - 1, int(parts[0]), float(parts[1]), float(parts[2])))
        cust_location.append( (float(parts[1]), float(parts[2])) )
        cust_demand.append( int(parts[0]) )

    # the depot is always the first customer in the input
    depot = customers[0]

    return depot, vehicle_count, vehicle_capacity, customers, cust_demand, cust_location


def input_data(file_path):
    with open(file_path, 'r') as input_data_file:
        input_data = input_data_file.read()

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])

    customers = []
    cust_location = []
    cust_demand = []

    for i in range(1, customer_count + 1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(i - 1, int(parts[0]), float(parts[1]), float(parts[2])))
        cust_location.append( (float(parts[1]), float(parts[2])) )
        cust_demand.append( int(parts[0]) )

    # the depot is always the first customer in the input
    depot = customers[0]

    return depot, vehicle_count, vehicle_capacity, customers, cust_demand, cust_location

def compute_euclidean_distance_matrix(data_points: list, dist_scale_param: int):
    distance_matrix = distance.cdist(data_points, data_points, 'euclidean') * dist_scale_param
    distances = []
    for row, loc in enumerate(data_points):
        distances.append(list(distance_matrix[row,:].astype(np.int16)))
    return distances

def output_solution(assignment, routing, manager, data):
    vehicle_tours = []
    dist_tours = []
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route_distance = 0
        route_load = 0
        customers_in_tour = []

        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            customers_in_tour.append(node_index)
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

        dist_tours.append( route_distance ) # + data['distance_matrix'][index][data["depot"]]


        vehicle_tours.append(customers_in_tour)
    total_dist = sum(dist_tours)

    return vehicle_tours, dist_tours, total_dist

def get_codes(verts):
    codes = []
    last_idx = len(verts) - 1
    for idx, vertex in enumerate(verts):
        if idx==0:
            codes.append(Path.MOVETO)
        elif idx==last_idx:
            codes.append(Path.CLOSEPOLY)
        else:
            codes.append(Path.LINETO)
    return codes


def get_verts(cust_location, tour):
    tour_locs = []
    x_max = 0.0
    x_min = 0.0
    y_max = 0.0
    y_min = 0.0
    for customer in tour:
        tour_locs.append(cust_location[customer])

        x = cust_location[customer][0]
        y = cust_location[customer][1]
        x_max = max(x, x_max)
        x_min = min(x, x_min)
        y_max = max(y, y_max)
        y_min = min(y, y_min)

    tour_locs.append(cust_location[0])  # Pad the return to the depot
    plot_lims = {}
    plot_lims["x_max"] = x_max
    plot_lims["x_min"] = x_min
    plot_lims["y_max"] = y_max
    plot_lims["y_min"] = y_min
    return tour_locs, plot_lims

def plot_vrp(vehicle_tours, cust_location, plot_lims_margin: float = 1.1, plot_lw: int =2, sum_tour_dist: float = 0.0):

    num_tours = len(vehicle_tours)
    color = cm.rainbow(np.linspace(0, 1, num_tours))
    x_max = 0.0
    x_min = 0.0
    y_max = 0.0
    y_min = 0.0

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for idx, tour in enumerate(vehicle_tours):
        verts, plot_lims = get_verts(cust_location, tour)
        codes = get_codes(verts)
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor="none", edgecolor=color[idx], lw=plot_lw)
        ax.add_patch(patch)
        ax.scatter(*zip(*verts), color=color[idx], s=plot_lw*5)
        x_max = max(plot_lims["x_max"], x_max)
        x_min = min(plot_lims["x_min"], x_min)
        y_max = max(plot_lims["y_max"], y_max)
        y_min = min(plot_lims["y_min"], y_min)

    ax.text(0.98, 0.98, 'Sum of tours\ndistances: {0:.2f}'.format(sum_tour_dist),
         horizontalalignment='right',
         verticalalignment='top',
         transform=ax.transAxes)
    ax.set_xlim(x_min * plot_lims_margin, x_max * plot_lims_margin)
    ax.set_ylim(y_min * plot_lims_margin, y_max * plot_lims_margin)
    plt.show()


def compute_total_tours_dist(vehicle_tours, cust_location, depot):
    distance_by_tour = []
    for tour in vehicle_tours:
        d = 0.0
        len_tour = len(tour)
        for idx, customer in enumerate(tour):
            if idx == 0:
                prev_cust = cust_location[customer]
            current_cust = cust_location[customer]
            d += distance.euclidean(prev_cust, current_cust)
            prev_cust = current_cust

            if idx == len_tour - 1:
                d += distance.euclidean(current_cust, cust_location[depot])

        distance_by_tour.append(d)
    return distance_by_tour
