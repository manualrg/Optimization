#!/usr/bin/python
# -*- coding: utf-8 -*-


def solve_it(input_data):

    from ortools.algorithms import pywrapknapsack_solver

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    values = []
    weights = []
    capacities = [capacity]
    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        _value = int(parts[0])
        _weight = int(parts[1])
        values.append(_value)
        weights.append(_weight)
    weights = [weights]

    # Instanciate the solver
    str_solver = 'NItems_{}_Capacity_{}'.format(item_count, capacity)
    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.
            KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, str_solver)

    solver.Init(values, weights, capacities)
    value = solver.Solve()

    packed_items = []
    packed_weights = []
    actual_cap = 0

    print('Total value =', value)
    for i in range(len(values)):
        if solver.BestSolutionContains(i):
            packed_items.append(i)
            packed_weights.append(weights[0][i])
            actual_cap += weights[0][i]


    # prepare the solution in the specified output format
    output_data = "Obj-value: {}, total weight: {} ({})\n".format(value, actual_cap, actual_cap/capacity)
    #output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, packed_items))
    return output_data


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python 01_knapsack_customDP.py ./data/ks_4_0)')

    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
    print(solve_it(input_data))
