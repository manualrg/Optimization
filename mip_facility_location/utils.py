from collections import namedtuple
from  scipy.spatial import distance
import numpy as np
import pandas as pd
import math

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])

def read_input_data(file_path):
    # Modify this code to run your optimization algorithm
    with open(file_path, 'r') as input_data_file:
        input_data = input_data_file.read()
    return input_data

def parse_data(input_data):

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])

    facilities = []
    fac_location = []
    fac_capacity = []
    fac_fixed_cost = []
    for i in range(1, facility_count + 1):
        parts = lines[i].split()
        facilities.append(Facility(i - 1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3]))))

        fac_location.append((float(parts[2]), float(parts[3])))
        fac_capacity.append(int(parts[1]))
        fac_fixed_cost.append(float(parts[0]))

    customers = []
    cust_location = []
    cust_demand = []

    for i in range(facility_count + 1, facility_count + 1 + customer_count):
        parts = lines[i].split()
        customers.append(Customer(i - 1 - facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

        cust_location.append((float(parts[1]), float(parts[2])))
        cust_demand.append(int(parts[0]))

    return facilities, facility_count, fac_location, fac_capacity, fac_fixed_cost, customers, customer_count, cust_location, cust_demand


def compute_solution_list(y, customers_count):
    solution = []
    for cust in range(customers_count):
        supplier = np.argmax(y[:, cust])
        solution.append(supplier)
    return solution

def model_data(facilities_data: dict, customers_data: dict):

    # Retrieve locations
    fac_loc = facilities_data["location"]
    cust_loc = customers_data["location"]

    # Get customers demand
    demand = customers_data["demand"]


    # Get facilities capacity and fixed cost
    capacities = facilities_data["capacity"]
    fixed_cost = facilities_data["fixed_cost"]

    # Get counts
    fac_num = facilities_data["count"]
    cust_num = customers_data["count"]

    dist_mat = get_distance_matrix(fac_loc, cust_loc)

    return dist_mat, capacities, fixed_cost, demand, fac_num, cust_num

def get_distance_matrix(facilities_loc: list, customers_loc: list):
    """
    Computes a distance matrix in euclidean space where rows represents facilities and columns are customerrs
    :param facilities_loc: list of tuples (x,y) with facilities cartesian coordinates
    :param customers_loc: list of tuples (x,y) with customers cartesian coordinates
    :return: np.ndarray
    """
    dist_mat = distance.cdist(facilities_loc, customers_loc, metric='euclidean', p=2)
    return dist_mat

def facility_to_cust_neighbour(distance_matrix: np.ndarray, facility_count: int, customer_count: int, k: int):
    """
    In order to build a sparse model, create some dicionaries to define decision variables regarding
    i and j pairs close enough
    :param distance_matrix: Distance matrix with facilities at rows and customers at columns
    :param facility_count: Number of facilities
    :param customer_count: Number of customers
    :param k: Each customer is linked with k closest facilities
    :return:
    :facility_closest_custs_n: A dictionary whose keys are facilities and values are a list of k closest custormers
    :cust_to_fac: A dictionary whose keys are customers and values are a list of suitable facilities
    """
    
    facility_closest_custs = {}
    for fac in range(0, facility_count):
        facility_closest_custs[fac] = []

    cust_to_fac = {}
    for cust in range(0, customer_count):
        cust_to_fac[cust] = []

    for cust in range(0, customer_count):
        closests_factories = pd.Series(distance_matrix[:, cust]).sort_values()[:k]
        for fac_idx, dist in closests_factories.items():
            facility_closest_custs[fac_idx].append(cust)
            cust_to_fac[cust].append(fac_idx)
    return facility_closest_custs, cust_to_fac
