import numpy as np
import pandas as pd
import random

from routing.tsp_local_search import utils as sa_utils, two_opt

### SA

def update_temp(T, alpha):
    return T * alpha


def SA_choice(delta, T):
    if delta < 0:  # Is an improvement because it decreases cost
        p = 1
    else:
        p = np.exp(-delta / T)  # Choose with probability e^(-Delta/T)
    return p


def SA_with_opt(tour, distance_matrix, depot, temperatures, param_cooling, n_restarts, seed: int = 123):
    random.seed(seed)
    temperatures.sort()
    min_temp = temperatures[0]
    max_temp = temperatures[1]
    new_tour = tour.copy()  # Updated when solution is chosen
    best_tour = tour.copy()  # Keep track of every iteration best solution
    history = []
    report = []
    for it in range(0, n_restarts):  # Cooling-Reheating cycles or restarts
        temperature = max_temp
        step = 0
        new_tour = best_tour  # Restart the cycle at the former iteration best tour
        new_cost = sa_utils.compute_tour_cost(new_tour, distance_matrix)
        min_cost = new_cost
        while temperature > min_temp:  # Cooling loop
            flg_choose = False
            # Randomly pick two nodes (except depot) and perform 2-opt swap
            [i, j] = random.sample([x for x in new_tour if x != depot], 2)
            if abs(j - i) == 1: continue  # Nodes already in the same edge, then skip
            neighbour_tour, delta_cost = two_opt.swap2opt(distance_matrix, new_tour, i, j)
            # if delta_cost<0: print(f"improved:{delta_cost}")
            # Choose tour with probability delta_E/T
            choose_prob = SA_choice(delta=delta_cost, T=temperature)
            flg_choose = choose_prob >= random.random()
            if flg_choose:
                new_tour = neighbour_tour.copy()
                new_cost += delta_cost
            step += 1
            # Update temperature cooling by a factor
            temperature = update_temp(T=temperature, alpha=param_cooling)
            # Keep track of best solution within each iteration
            if new_cost < min_cost:
                min_cost = new_cost
                best_tour = new_tour
            history.append((it, step, temperature, i, j, delta_cost, choose_prob, flg_choose, new_cost, min_cost))
        report.append((it, best_tour, min_cost))
    history_df = pd.DataFrame(data=history, columns=["iteration", "step",
                                                     "temperature", "node_u", "node_v",
                                                     "test_cost", "probability", "flg_choose",
                                                     "step_cost", "iteration_cost"])
    report_df = pd.DataFrame(data=report, columns=["iteration", "tour", "cost"])
    print("==> Simulated Annealing iteration summary:")
    print(history_df.groupby("iteration")["flg_choose", "step_cost"] \
          .agg({"flg_choose": np.mean, "step_cost": np.min}))
    return best_tour, history_df, report_df