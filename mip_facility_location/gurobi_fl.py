import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *


def model_sparse(dist_mat, capacities, fixed_cost, demand, fac_num, cust_num, alpha: float = 1.0, time_limit_secs: float =100.0):
    # Define model
    m = Model()

    # Instanciate variables as dictionaries
    x = {}
    y = {}

    c = {}  # Distance matrix (not a variable)

    # Populate variables
    for i in range(fac_num):
        x[i] = m.addVar(vtype=GRB.BINARY, name=f'x{i}')
    for i in range(fac_num):
        for j in range(cust_num):
            y[(i, j)] = m.addVar(vtype=GRB.BINARY, name=f'y{j},{i}')
            c[(i, j)] = alpha * dist_mat[i, j]  # Not a model variable

    # Add variables to de model
    m.update()

    # Add constraints
    for i in range(fac_num):
        m.addConstr(lhs=quicksum(y[(i, j)] * demand[j] for j in range(cust_num)),
                    sense=GRB.LESS_EQUAL,
                    rhs=capacities[i] * x[i],
                    name=f'capacity_demand_{i}')
        m.addConstr(lhs=quicksum(y[i, j] for j in range(cust_num)),
                    sense=GRB.GREATER_EQUAL,
                    rhs=x[i],
                    name=f'matching_{i}')  # if xi=0 then its correspondent row must add up to 0

    for j in range(cust_num):
        m.addConstr(lhs=quicksum(y[(i, j)] for i in range(fac_num)),
                    sense=GRB.EQUAL,
                    rhs=1,
                    name=f'single_supplier_{j}')

    # Define objective function
    m.setObjective(quicksum(
                        fixed_cost[i] * x[i] + quicksum(c[(i, j)] * y[(i, j)]
                                                        for i in range(fac_num)) for j in range(cust_num)
                        )
                    )
    # Set solver time limit in seconds
    m.setParam('TimeLimit', time_limit_secs)

    # Solve!
    m.optimize()

    return m


def model_dense(dist_mat, capacities, fixed_cost, demand, fac_num, cust_num, alpha: float = 1.0):
    # Define model
    m = Model()

    # Instanciate variables as dictionaries
    x = {}
    y = {}

    c = {}  # Distance matrix (not a variable)

    # Populate variables
    for i in range(1, fac_num+1):
        x[i] = m.addVar(vtype=GRB.BINARY, name=f'x{i}')
    for j in range(cust_num):
        y[j] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=fac_num, name=f'y{j}')

    for i in range(fac_num):
        for j in range(cust_num):
            c[(i, j)] = alpha * dist_mat[i, j]  # Not a model variable

    # Add variables to de model
    m.update()

    # Add constraints
    for i in range(fac_num):
        m.addConstr(lhs=quicksum((y[j] == i) * demand[j] for j in range(cust_num)),
                    sense=GRB.LESS_EQUAL,
                    rhs=capacities[i] * x[i],
                    name=f'capacity_demand_{i}')

    # Define objective function
    m.setObjective(quicksum(
                        fixed_cost[i] * x[i] + quicksum(c[(i, j)] * (y[j] == i)
                                                        for i in range(fac_num)) for j in range(cust_num)
                        )
                    )

    # Solve!
    m.optimize()

    return m



def report_grb_dc(m, fac_num, cust_num):
    grb_vars = {}
    grb_x_vars = []
    grb_y_vars = []
    for v in m.getVars():
        name = v.varName
        value = v.x
        grb_vars[f'{name}'] = value
        if name.startswith("x"):
            grb_x_vars.append(value)
        elif name.startswith("y"):
            grb_y_vars.append(value)
        else:
            print(f'There are decision variables in the model not preffixed by x or y: {name}')

    solution_x = np.array(grb_x_vars)
    solution_y = np.reshape(np.array(grb_y_vars), (fac_num, cust_num))
    return solution_x, solution_y