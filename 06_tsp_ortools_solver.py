#!/usr/bin/python
# -*- coding: utf-8 -*-

def solve_it(input_data):

    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    from routing.tsp_ortools import utils, ortools_tsp

    DIST_SCALE_FACTOR = 1000
    # Parse and prepare input data
    points = utils.parse_data(input_data)
    dist_mat_dict,_ = utils.compute_euclidean_distance_matrix(points, DIST_SCALE_FACTOR)
    data = utils.create_data_model(dist_mat_dict)

    # Select algorithm
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 30
    search_parameters.log_search = False

    #Solver problem
    manager, routing, assignment = ortools_tsp.model_tsp(data, search_parameters)

    #Report solutin
    obj, solution, route_arcs = utils.output_solution(manager, routing, assignment, DIST_SCALE_FACTOR)

    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

