#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def solve_it(input_data):

    from mip_facility_location import utils, ortools_fl as ort_fl
    # parse the input
    facilities, facility_count, fac_location, fac_capacity, fac_fixed_cost, customers, customer_count, cust_location, cust_demand = utils.parse_data(input_data)

    facilities_data = {}
    facilities_data["location"] = fac_location
    facilities_data["capacity"] = fac_capacity
    facilities_data["fixed_cost"] = fac_fixed_cost
    facilities_data["count"] = facility_count

    customers_data = {}
    customers_data["location"] = cust_location
    customers_data["demand"] = cust_demand
    customers_data["count"] = customer_count

    dist_mat, capacities, fixed_cost, demand, fac_num, cust_num = utils.model_data(facilities_data, customers_data)
    facility_closest_custs_n, cust_to_fac = utils.facility_to_cust_neighbour(dist_mat, facility_count, customer_count, int(facility_count/10))
    model, _, dc_vars = ort_fl.model_sparse_partial(dist_mat, facility_closest_custs_n, cust_to_fac, capacities, fixed_cost, demand, fac_num,
                          1.0, 30)
    x = dc_vars["x"]
    y = dc_vars["y"]

    x = ort_fl.report_ortools_dc(x, (fac_num))
    y = ort_fl.report_ortools_dc(y, (fac_num, cust_num))

    solution = utils.compute_solution_list(y, cust_num)


    # calculate the cost of the solution
    obj = sum([f.setup_cost * x[f.index] for f in facilities])
    for customer in customers:
        obj += length(customer.location, facilities[solution[customer.index]].location)

    # prepare the solution in the specified output format
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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python 05_facility_location_mip_gurobi_solver.py ./data/fl_16_2)')

