from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

def input_data_reader(path :str):
    import sys
    if len(path) > 1:
        file_location = path.strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python 01_knapsack_customDP.py ./data/ks_4_0)')
    return input_data

def input_data_parser(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(
            Item(i-1, int(parts[0]), int(parts[1]))
        )

    return items, capacity, item_count

def format_input_data(items, prepend):
    new_items = [prepend]
    for item in items:
        new_items.append(Item(item.index + 1, item.value, item.weight))
    return new_items
