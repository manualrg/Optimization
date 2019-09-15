# Optimization algorithms and techologies repository
------
manualrg

## Getting Started
This project is aimed at experimenting and assessing several optimization algorithms through a set of problems.
These problems are extracted from Discrete Optimization course on Coursera, they are a curated list of the most interesting problems in the topic.

The structure of this repository is the following:
* Dynamic programming to solve the knapsack problem: Discrete binning
* Constraint programming applied to a graph colouring problem: Assign colour to each node given that colours must not be repeated in neighbour nodes.
* MIP: Use ORTools CBC solver and Gurobi MILP solver in order to solve a Capacitated Facility Location problem
* Local search: Solve the TSP and VRP using local search algorithm (2-opt), metaheuristics and ORTools


In the main folder, there are few .py examples and several  notebooks that run examples.
The examples are placed in each section folder, for example:

```.\mip_facility_location```

Contains both data and several packages used in this examples.

```.\mip_facility_location\data```

Few example problems are placed in \data folder

```
.\mip_facility_location\gurobi_fl.py
.\mip_facility_location\ortools_fl.py
.\mip_facility_location\utils.py
```
Apart from data, several packages are stored, so that they can be called from the examples related to mip_falitiy_location in the main folder:

```
./04_facility_location_mip_ortools_solver.py
./04_facility_location_mip_ortools_solver.ipynb
./04_facility_location_mip_gurobi_solver.py
```

In general, .py examples just run and solver the problem, whereas notebooks explore input data, compare several strategies and/or visualize results.


## Prerequisites
Common Python packages are used:

* Python 3.7
* Numpy 1.16
* Scipy 1.3
* Pandas 0.25
* Matplotlib 3.1
* Networkx 2.3


## Installing
One of the main aims of this project is to test several Optimization technologies, such us ORTools and Gurobi.
* ortools (Open)
```
pip install --upgrade --user ortools```https://developers.google.com/optimization/install/
```
* Gurobi (License required, there are several free options: https://www.gurobi.com/es/products/licensing-options/)
1. Quickstart guide: https://www.gurobi.com/resource/starting-with-gurobi/
3. Install Python interface:
```
pip install gurobipy 
```

# Running examples
Just run in cmd the following sentece:
* Call the solver: It will read and parse input data, model the optimization problem, solve it, and print results
* Select input data
```
python ./04_facility_location_mip_ortools_solver.py .\mip_facility_location\data\fl_25_2
```

# Comments
If you are interested in the topic, I strongly recommend joining Discrete Optimization course on Coursera and develop your own algorithms, this techologies or other ones (like SCIP) and takle this challenging course.
I hope that yout find this material useful, enjoy!
