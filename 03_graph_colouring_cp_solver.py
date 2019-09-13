#!/usr/bin/python
# -*- coding: utf-8 -*-


def solve_it(input_data):
    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    from cp_graph_colouring import cp_search_prune as cp, input
    import networkx as nx

    g = input.create_graph(edges)
    nodes = list(range(0, node_count))
    nodes_df = input.get_node_data(g, nodes)

    solution = [-1] * node_count

    domain, colours_bank = cp.create_domain(int(node_count/2), node_count)

    solution, colours_bank, domain, report_df = \
        cp.dynamic_search(g=g, nodes_df=nodes_df,
                       domain=domain, solution=solution, colours_bank=colours_bank,
                       param_dyn_search=0.2, dev=False)

    # prepare the solution in the specified output format
    output_data = str(node_count) + ' ' + str(0) + '\n'
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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

