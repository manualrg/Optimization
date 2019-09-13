from ortools.linear_solver import pywraplp
import numpy as np

def model_sparse(dist_mat: np.ndarray, capacities: list, fixed_cost: list, demand: list,
                 fac_num: int, cust_num: int, alpha: float = 1.0, max_time: float = 30.0):
    """
    Defines a MIP problem with the following scheme:
    x[i] = {0,1} whether facility i is opened
    y[i,j] = {0,1} whether facility i supplies customer j
    min{ sum(x[i]*fixed_cost[i]) + sum(y[i]*alpha)}
    s.t.
    (I) A facility cannot supply more than its capacity
    (II) each customer must be supplied by a single facility
    :param dist_mat: Distance matrix with facilities at rows and customers at columns
    :param capacities: List of facilities capacities
    :param fixed_cost: List facilities fixed costs figures
    :param demand: List of customers demand
    :param fac_num: Number of facilities
    :param cust_num: Number of customers
    :param alpha: Conversion factor from distance to cost (e.g. €/km)
    :param max_time: Max running time in seconds
    :return: solver, objective value, decision variables
    """
    # Define model
    # Create the mip solver with the CBC backend.
    solver = pywraplp.Solver('capacitated_facility_assignment',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    # Instanciate variables as dictionaries
    x = {}
    y = {}

    c = {}  # Distance matrix (not a variable)

    # Populate variables
    for i in range(fac_num):
        x[i] = solver.IntVar(0, 1, f'x{i}')
    for i in range(fac_num):
        for j in range(cust_num):
            y[(i, j)] = solver.IntVar(0, 1, f'y{i},{j}')
            c[(i, j)] = alpha * dist_mat[i, j]  # Not a model variable

    print('Number of variables =', solver.NumVariables())
    dc_vars = {}
    dc_vars["x"] = x
    dc_vars["y"] = y

    # Add constraints
    for i in range(fac_num):
        solver.Add(sum(y[(i, j)] * demand[j] for j in range(cust_num))
                   <=
                   capacities[i] * x[i],
                   f'capacity_demand_{i}')
    for j in range(cust_num):
        solver.Add(sum(y[(i, j)] for i in range(fac_num))
                   ==
                   1,
                   f'single_supplier_{j}')

    print('Number of constraints =', solver.NumConstraints())

    # Define objective function
    solver.Minimize(sum(fixed_cost[i] * x[i] for i in range(fac_num))
                    + sum(c[(i, j)] * y[(i, j)] for i in range(fac_num) for j in range(cust_num))
                    )
    # Set solver max time in seconds
    if max_time<=0:
        print(f"Maxium solver time is less or equal to zero {max_time}, therefore, no limit is establish")
    else:
        solver.set_time_limit(int(max_time*1000))
    # Solve!
    result_status = solver.Solve()
    # Check optimality

    if result_status == pywraplp.Solver.OPTIMAL: print("Optimal!")
    elif result_status == pywraplp.Solver.FEASIBLE: print("Potentially suboptimal")


    assert solver.VerifySolution(1e-7, True)
    obj = solver.Objective().Value()
    print('Objective value =', solver.Objective().Value())

    return solver, obj, dc_vars

def model_sparse_partial(dist_mat: np.ndarray, facility_closest_custs_n: dict, cust_to_fac: dict,
                         capacities: list, fixed_cost: list, demand: list,
                         fac_num: int, alpha: float = 1.0, max_time: float = 30.0):
    """
    Defines a MIP problem with the following scheme:
    x[i] = {0,1} whether facility i is opened
    y[i,j] = {0,1} whether facility i supplies customer j
    min{ sum(x[i]*fixed_cost[i]) + sum(y[i]*alpha)}
    s.t.
    (I) A facility cannot supply more than its capacity
    (II) each customer must be supplied by a single facility
    Decision variables are only define for i,j pairs defined by: facility_closest_custs_n and cust_to_fac
    :param dist_mat: Distance matrix with facilities at rows and customers at columns
    :param facility_closest_custs_n: A dictionary whose keys are facilities and values are a list of k closest custormers
    :param cust_to_fac: A dictionary whose keys are customers and values are a list of suitable facilities
    :param capacities: List of facilities capacities
    :param fixed_cost: List facilities fixed costs figures
    :param demand: List of customers demand
    :param fac_num: Number of facilities
    :param cust_num: Number of customers
    :param alpha: Conversion factor from distance to cost (e.g. €/km)
    :param max_time: Max running time in seconds
    :return: solver, objective value, decision variables
    """
    # Define model
    # Create the mip solver with the CBC backend.
    solver = pywraplp.Solver('capacitated_facility_assignment',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    # Instanciate variables as dictionaries
    x = {}
    y = {}

    c = {}  # Distance matrix (not a variable)

    # Populate variables
    for i in range(fac_num):
        x[i] = solver.IntVar(0, 1, f'x{i}')
    for i, customers in facility_closest_custs_n.items():
        for j in customers:
            y[(i, j)] = solver.IntVar(0, 1, f'y{i},{j}')
            c[(i, j)] = alpha * dist_mat[i, j]  # Not a model variable

    print('Number of variables =', solver.NumVariables())
    dc_vars = {}
    dc_vars["x"] = x
    dc_vars["y"] = y

    # Add constraints
    for i, customers in facility_closest_custs_n.items():
        solver.Add(sum(y[(i, j)] * demand[j] for j in customers)
                   <=
                   capacities[i] * x[i],
                   f'capacity_demand_{i}')
        # solver.Add(sum(y[(i, j)] for j in customers)
        #            >=
        #            x[i],
        #            f'matching_{i}')

    for j, facilities in cust_to_fac.items():
        solver.Add(sum(y[(i, j)] for i in facilities)
                   ==
                   1,
                   f'single_supplier_{j}')

    print('Number of constraints =', solver.NumConstraints())

    # Define objective function
    solver.Minimize(sum(fixed_cost[i] * x[i] for i in range(fac_num))
                    + sum(c[(i, j)] * y[(i, j)] for i, customers in facility_closest_custs_n.items() for j in customers)
                    )
    # Set solver max time in seconds
    if max_time<=0:
        print(f"Maxium solver time is less or equal to zero {max_time}, therefore, no limit is establish")
    else:
        solver.set_time_limit(int(max_time*1000))
    # Solve!
    result_status = solver.Solve()
    # Check optimality
    # assert (result_status == pywraplp.Solver.OPTIMAL or result_status == pywraplp.Solver.FEASIBLE)
    if result_status == pywraplp.Solver.OPTIMAL: print("Optimal!")
    elif result_status == pywraplp.Solver.FEASIBLE: print("Potentially suboptimal")

    # The solution looks legit (when using solvers others than
    # GLOP_LINEAR_PROGRAMMING, verifying the solution is highly recommended!).
    assert solver.VerifySolution(1e-7, True)
    obj = solver.Objective().Value()
    print('Objective value =', solver.Objective().Value())

    return solver, obj, dc_vars


def report_ortools_dc(dc_var: dict, dc_shape: tuple):
    """
    Collects decision variables data for reporting purposes. DCs are integers
    :param dc_var: A dictionary containing model's decisions variables in ortools
    :param dc_shape: A tuple with output dimensions
    :return: An ndarray containting decision variables values of the solution
    """
    solution = np.zeros(dc_shape)
    for key, value in dc_var.items():
        solution[key] = int(value.solution_value())

    return solution