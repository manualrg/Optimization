#!/usr/bin/python
# -*- coding: utf-8 -*-


def solve_it(input_data, capacity_step):

    from collections import namedtuple
    Item = namedtuple("Item", ['index', 'value', 'weight'])
    from dp_knapsack import dynamic_programming as dp
    from dp_knapsack import input as input

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        _value = int(parts[0])
        _weight = int(parts[1])
        items.append(
            Item(i-1, _value, _weight)
        )

    items_pad = input.format_input_data(items, Item(0, 0, 0))

    matrix = dp.vec_compute_dp_table_step(items=items_pad,
                                  capacity=capacity, capacity_step=capacity_step, item_count=item_count)
    value = matrix[-1, -1]
    taken = dp.index_items_step(results_mat=matrix, items=items_pad, obj_value=value,
                                 item_count=item_count, capacity_step=capacity_step)
    actual_cap = dp.knapshack_check_capacity(items, taken)

    # prepare the solution in the specified output format
    output_data = "Obj-value: {}, total weight: {} ({})\n".format(value, actual_cap, actual_cap/capacity)
    #output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 2:
        file_location = sys.argv[1].strip()
        capacity_step = int(sys.argv[2].strip())
    elif len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        capacity_step = 1
    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python 01_knapsack_customDP.py ./data/ks_4_0)')

    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
    print(solve_it(input_data, capacity_step))
