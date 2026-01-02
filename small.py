
from pyscipopt import Model, quicksum

# ------------------- Data -------------------
jobs  = [0, 1, 2]
nodes = [0, 1]

durations = [3, 2, 4]
loads     = [2, 1, 3]
capacities = [4, 4]

precedences = [(0, 2)]
comm_delay = [[0, 1],
              [1, 0]]

HORIZON = 1000   # safe big-M

# ------------------- Model -------------------
model = Model("Scheduling_Small")
model.hideOutput(False)  # remove if you want silence

# Variables
x = {(i,j): model.addVar(vtype="B", name=f"x_{i}_{j}") for i in nodes for j in jobs}
s = {j: model.addVar(vtype="C", lb=0, name=f"s_{j}") for j in jobs}
Cmax = model.addVar(vtype="C", name="Cmax")

# Sequencing variables y[i,j,k] = 1 if job j before k on node i (j < k)
y = {}
for i in nodes:
    for j1 in jobs:
        for j2 in jobs:
            if j1 < j2:
                y[i,j1,j2] = model.addVar(vtype="B", name=f"y_{i}_{j1}_{j2}")

# ------------------- Constraints -------------------

# 1. Assignment
for j in jobs:
    model.addCons(quicksum(x[i,j] for i in nodes) == 1)

# 2. Capacity
for i in nodes:
    model.addCons(quicksum(loads[j] * x[i,j] for j in jobs) <= capacities[i])

# 3. Non-overlap (big-M) - exactly as in the report
for i in nodes:
    for j1 in jobs:
        for j2 in jobs:
            if j1 < j2:
                model.addCons(s[j2] >= s[j1] + durations[j1] - HORIZON*(1 - y[i,j1,j2]) - HORIZON*(2 - x[i,j1] - x[i,j2]))
                model.addCons(s[j1] >= s[j2] + durations[j2] - HORIZON*    y[i,j1,j2]  - HORIZON*(2 - x[i,j1] - x[i,j2]))

# 4. CORRECT Precedence + Communication Delay (fixed)
for pred, succ in precedences:
    for i in nodes:
        for ip in nodes:
            z = model.addVar(vtype="B", name=f"z_{i}_{pred}_{ip}_{succ}")
            # linking z to assignments
            model.addCons(z <= x[i,pred])
            model.addCons(z <= x[ip,succ])
            model.addCons(z >= x[i,pred] + x[ip,succ] - 1)
            # conditional precedence: active only when z == 1
            # when z==1: s_succ >= s_pred + dur + comm_delay
            # when z==0: RHS becomes s_pred + dur - HORIZON (very loose)
            model.addCons(s[succ] >= s[pred] + durations[pred] + comm_delay[i][ip] * z
                          - HORIZON * (1 - z))

# 5. Makespan
for j in jobs:
    model.addCons(Cmax >= s[j] + durations[j])

# ------------------- Solve -------------------
model.setObjective(Cmax, "minimize")
model.optimize()

# ------------------- Output -------------------
if model.getStatus() == "optimal":
    print(f"\nOPTIMAL MAKESPAN = {model.getObjVal()}\n")
    for j in jobs:
        for i in nodes:
            if model.getVal(x[i,j]) > 0.5:
                st = model.getVal(s[j])
                print(f"Job {j} → Node {i} | start = {st:.1f} | end = {st + durations[j]:.1f}")
else:
    print("Status:", model.getStatus())
