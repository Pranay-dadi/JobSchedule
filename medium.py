from ortools.sat.python import cp_model
import random

# -------------------------------
# Problem Data (medium instance)
# -------------------------------
n_jobs = 12
m_nodes = 4

jobs = list(range(n_jobs))
nodes = list(range(m_nodes))

# Random job durations and loads
random.seed(42)
durations = [random.randint(1, 10) for _ in jobs]
loads = [random.randint(1, 5) for _ in jobs]

# Node capacities
capacities = [15, 15, 15, 15]

# Random precedence constraints (10% of pairs)
precedences = [(i, j) for i in jobs for j in jobs if i != j and random.random() < 0.1]

# Communication delays (0 if same node, 1-3 otherwise)
comm_delay = [[0 if i == j else random.randint(1, 3) for j in nodes] for i in nodes]

M = 1000  # Big-M constant

# -------------------------------
# CP-SAT Model
# -------------------------------
model = cp_model.CpModel()

# Decision Variables
x = {(i, j): model.NewBoolVar(f'x_{i}_{j}') for i in nodes for j in jobs}
s = {j: model.NewIntVar(0, sum(durations) * 10, f'start_{j}') for j in jobs}
Cmax = model.NewIntVar(0, sum(durations) * 10, 'Cmax')
y = {(i, j, k): model.NewBoolVar(f'y_{i}_{j}_{k}') 
     for i in nodes for j in jobs for k in jobs if j < k}

# -------------------------------
# Constraints
# -------------------------------

# Assignment
for j in jobs:
    model.Add(sum(x[i, j] for i in nodes) == 1)

# Capacity
for i in nodes:
    model.Add(sum(loads[j] * x[i, j] for j in jobs) <= capacities[i])

# Sequencing constraints
for i in nodes:
    for j in jobs:
        for k in jobs:
            if j < k:
                # Only enforce sequencing if both assigned
                model.Add(y[i, j, k] <= x[i, j])
                model.Add(y[i, j, k] <= x[i, k])
                model.Add(s[k] >= s[j] + durations[j] - M * (1 - y[i, j, k]) - M * (2 - x[i, j] - x[i, k]))
                model.Add(s[j] >= s[k] + durations[k] - M * y[i, j, k] - M * (2 - x[i, j] - x[i, k]))

# Precedence constraints with communication delay
for k, l in precedences:
    for i in nodes:
        for i_prime in nodes:
            # Auxiliary variable for x[i,k] * x[i_prime,l]
            z = model.NewBoolVar(f'z_{i}_{k}_{i_prime}_{l}')
            # Linearize the product
            model.Add(z <= x[i, k])
            model.Add(z <= x[i_prime, l])
            model.Add(z >= x[i, k] + x[i_prime, l] - 1)
            # Precedence constraint with delay
            model.Add(s[l] >= s[k] + durations[k] + comm_delay[i][i_prime] * z)

# Makespan
for j in jobs:
    model.Add(Cmax >= s[j] + durations[j])

# Objective
model.Minimize(Cmax)

# -------------------------------
# Solver Parameters
# -------------------------------
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60  # limit runtime
solver.parameters.num_search_workers = 8    # parallel search

status = solver.Solve(model)

# -------------------------------
# Print Solution
# -------------------------------
if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print(f'Minimum makespan: {solver.Value(Cmax)}\n')
    for j in jobs:
        assigned_node = [i for i in nodes if solver.Value(x[i, j]) == 1][0]
        print(f'Job {j} assigned to Node {assigned_node}, start at {solver.Value(s[j])}, duration {durations[j]}')
else:
    print('No feasible solution found within time limit.')

# Print problem data for reference
print(f'\nProblem Data:')
print(f'Durations: {durations}')
print(f'Loads: {loads}')
print(f'Precedences: {precedences}')
print(f'Communication Delays: {comm_delay}')
