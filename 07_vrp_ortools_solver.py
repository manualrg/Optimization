#!/usr/bin/python
# -*- coding: utf-8 -*-


def solve_it(input_data):
    # Modify this code to run your optimization algorithm
    from routing.vrp_ortools import utils, vrp_ortools as vrp
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp

    # Parse the input
    depot, vehicle_count, vehicle_capacity, customers, cust_demand, cust_location = utils.parse_data(input_data)
    # Model data
    data = {}

    data['num_vehicles'] = vehicle_count
    data['depot'] = 0
    data["vehicle_capacities"] = [vehicle_capacity] * vehicle_count

    data["demands"] = cust_demand
    data["distance_matrix"] = utils.compute_euclidean_distance_matrix(data_points=cust_location, dist_scale_param=1)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING)
    search_parameters.time_limit.seconds = 180

    assignment, routing, manager = vrp.cvrp_model(data, search_parameters, 3000, 100)
    vehicle_tours, dist_tours, dist_est = utils.output_solution(assignment, routing, manager, data)
    distance_by_tour = utils.compute_total_tours_dist(vehicle_tours, cust_location, data['depot'])
    obj = sum(distance_by_tour)

    depot = customers[0]
    outputData = '%.2f' % obj + ' ' + str(0) + '\n'
    for v in range(0, vehicle_count):
        outputData += ' '.join([str(customer) for customer in vehicle_tours[v]]) + ' ' + str(depot.index) + '\n'

    return outputData


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:

        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

