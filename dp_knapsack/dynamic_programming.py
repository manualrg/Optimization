import numpy as np
from collections import namedtuple
from operator import attrgetter
from collections import namedtuple

##################### Dynammic programming: Knapsack problem #####################
# Given a set of items, that each one has a weight and a value, fill a knapsack with a limited capacity
# maximizing its value. Items are not divisible


Item = namedtuple("Item", ['index', 'value', 'weight'])

def solver_greedy1(items, capacity):
    # a trivial greedy algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    value = 0
    weight = 0
    taken = [0] * len(items)

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight

    return taken, value


def solver_greedy2(items, capacity):
    from operator import attrgetter
    value = 0
    weight = 0
    taken = [0] * len(items)
    items_srt = sorted(items, key=attrgetter('specific_value'))

    for item in items_srt:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight
    return taken, value

def mat_O(k: int, j: int, v_j: int, w_j: int, matrix: np.ndarray):
    """
    For each step, decide whether the item fits or not
    and in the former case, if it must be included.

    Returns the value to be inserted in the DP table
    """
    former = matrix[k, j - 1]
    if w_j <= k:
        new = v_j + matrix[k - w_j, j - 1]
        res = max(former, new)
    else:
        res = former
    return res

def vec_compute_dp_table(items: list, capacity: int, item_count: int):
    """Computes a DP table as a 2darray (Numpy) and finds the objective value"""
    n_rows, n_cols = capacity + 1, item_count + 1
    matrix = np.zeros((n_rows, n_cols), dtype=np.int32)
    for col_idx, item in enumerate(items):
        j = item.index
        w_j = item.weight
        v_j = item.value
        if col_idx > 0:
            col_prev = matrix[:, col_idx - 1].copy()
            col_prev_split = np.split(col_prev, [w_j])
            col_prev_1, col_prev_2 = col_prev_split[0], col_prev_split[1]
            col_prev_2_lag = (np.roll(col_prev, w_j) + v_j)[w_j:]
            col_curr = np.maximum(col_prev_2, col_prev_2_lag)
            col_curr = np.concatenate((col_prev_1, col_curr))
            matrix[:, col_idx] = col_curr.copy()
    return matrix

def vec_compute_dp_table_step(items: list,
                              capacity: int, capacity_step: int, item_count: int):
    """Computes a DP table as a 2darray (Numpy) and finds the objective value"""
    n_rows, n_cols = int(np.ceil(capacity / capacity_step)) + 1, item_count + 1
    matrix = np.zeros((n_rows, n_cols), dtype=np.int32)
    for col_idx, item in enumerate(items):
        j = item.index
        w_j = item.weight
        adj_w_j = int(np.ceil((w_j / capacity_step)))
        v_j = item.value
        if col_idx > 0:
            col_prev = matrix[:, col_idx - 1].copy()
            col_prev_split = np.split(col_prev, [adj_w_j])
            col_prev_1, col_prev_2 = col_prev_split[0], col_prev_split[1]
            col_prev_2_lag = (np.roll(col_prev, adj_w_j) + v_j)[adj_w_j:]
            col_curr = np.maximum(col_prev_2, col_prev_2_lag)
            col_curr = np.concatenate((col_prev_1, col_curr))
            matrix[:, col_idx] = col_curr.copy()
    return matrix

def index_items(results_mat: np.ndarray, items: list, obj_value: int, item_count: int):
    """Backward analyzes the resulting DP table to determine what items are chosen"""
    n_rows, n_cols = np.shape(results_mat)
    current_target_val = obj_value
    current_target_idx = n_rows - 1
    taken = [0] * item_count

    # Iterate matrix columns reversely, skipping the most right one, that holds the objective value
    for col_idx in range(n_cols - 1, 0, -1):
        # Determinine next column target cell value
        next_target_val = results_mat[current_target_idx, col_idx - 1]
        # Check whether next target cell value is equal or lower than current
        if next_target_val == current_target_val:
            # If equals, keep on the same row
            next_target_idx = current_target_idx

        else:
            # If next is lower than current, substract item weight
            next_target_idx = current_target_idx - items[col_idx].weight
            next_target_val = results_mat[next_target_idx, col_idx - 1]

            current_target_idx = next_target_idx
            current_target_val = next_target_val

            taken[col_idx - 1] = 1

    return taken


def index_items_step(results_mat: np.ndarray, items: list, obj_value: int, item_count: int, capacity_step: int):
    """Backward analyzes the resulting DP table to determine what items are chosen"""
    n_rows, n_cols = np.shape(results_mat)
    current_target_val = obj_value
    current_target_idx = n_rows - 1
    taken = [0] * item_count

    # Iterate matrix columns reversely, skipping the most right one, that holds the objective value
    for col_idx in range(n_cols - 1, 0, -1):
        # Determinine next column target cell value
        next_target_val = results_mat[current_target_idx, col_idx - 1]
        # Check whether next target cell value is equal or lower than current
        if next_target_val == current_target_val:
            # If equals, keep on the same row
            next_target_idx = current_target_idx

        else:
            # If next is lower than current, substract item weight
            w_adj = int(np.ceil(items[col_idx].weight / capacity_step))
            next_target_idx = current_target_idx - w_adj
            next_target_val = results_mat[next_target_idx, col_idx - 1]

            current_target_idx = next_target_idx
            current_target_val = next_target_val

            taken[col_idx - 1] = 1

    return taken


def knapshack_check_capacity(items, taken):
    s = 0
    for item, flg_taken in zip(items, taken):
        if flg_taken == 1:
            s += item.weight
    return s



