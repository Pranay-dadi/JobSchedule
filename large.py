import random

# -------------------------------
# Problem Data (large instance)
# -------------------------------
n_jobs = 100
m_nodes = 6

jobs = list(range(n_jobs))
nodes = list(range(m_nodes))

random.seed(123)

# Job durations and loads
durations = [random.randint(1, 10) for _ in jobs]
loads = [random.randint(1, 5) for _ in jobs]

# Node capacities
capacities = [30, 30, 30, 30, 30, 30]

# Random precedence constraints (10% probability)
precedences = [(i,j) for i in jobs for j in jobs if i != j and random.random() < 0.1]

# Communication delays (0 if same node, else 1-3)
comm_delay = [[0 if i==j else random.randint(1,3) for j in nodes] for i in nodes]

# -------------------------------
# Genetic Algorithm Parameters
# -------------------------------
population_size = 50
generations = 200
mutation_rate = 0.1

# -------------------------------
# Helper Functions
# -------------------------------

def initialize_population():
    population = []
    for _ in range(population_size):
        # Random assignment: job -> node
        assignment = [random.choice(nodes) for _ in jobs]
        population.append(assignment)
    return population

def compute_makespan(assignment):
    # Schedule jobs on each node
    node_schedules = {i: [] for i in nodes}
    job_start_times = [0]*n_jobs

    # Build schedule respecting precedence and communication
    unscheduled = set(jobs)
    ready = [j for j in jobs if all(k not in unscheduled for k,_ in precedences if _==j)]
    
    while unscheduled:
        if not ready:  # cyclic or blocked
            ready = list(unscheduled)
        for j in ready:
            node = assignment[j]
            # Start time: max(prev jobs on same node + comm delays + duration)
            prev_jobs = node_schedules[node]
            start_time = 0
            if prev_jobs:
                last_job = prev_jobs[-1]
                start_time = job_start_times[last_job] + durations[last_job]
            # Check precedence
            for k,l in precedences:
                if l==j:
                    pred_node = assignment[k]
                    start_time = max(start_time, job_start_times[k] + durations[k] + comm_delay[pred_node][node])
            job_start_times[j] = start_time
            node_schedules[node].append(j)
        unscheduled -= set(ready)
        ready = [j for j in unscheduled if all(k not in unscheduled for k,_ in precedences if _==j)]
    
    # Makespan is max completion time
    completion_times = [job_start_times[j]+durations[j] for j in jobs]
    return max(completion_times), job_start_times

def crossover(parent1, parent2):
    point = random.randint(1, n_jobs-1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(individual):
    for j in range(n_jobs):
        if random.random() < mutation_rate:
            individual[j] = random.choice(nodes)
    return individual

# -------------------------------
# GA Main Loop
# -------------------------------
population = initialize_population()

best_solution = None
best_makespan = float('inf')

for gen in range(generations):
    # Evaluate fitness
    fitness = []
    for ind in population:
        makespan, _ = compute_makespan(ind)
        fitness.append((makespan, ind))
    fitness.sort(key=lambda x: x[0])
    
    # Update best
    if fitness[0][0] < best_makespan:
        best_makespan = fitness[0][0]
        best_solution = fitness[0][1]
    
    # Selection (top 50%)
    selected = [ind for _,ind in fitness[:population_size//2]]
    
    # Crossover + Mutation
    next_population = []
    while len(next_population) < population_size:
        parents = random.sample(selected, 2)
        child1, child2 = crossover(parents[0], parents[1])
        next_population.append(mutate(child1))
        if len(next_population) < population_size:
            next_population.append(mutate(child2))
    
    population = next_population

# -------------------------------
# Print Best Solution
# -------------------------------
print(f'Best makespan found: {best_makespan}')
job_starts = compute_makespan(best_solution)[1]
for j in range(n_jobs):
    print(f'Job {j} assigned to Node {best_solution[j]}, start at {job_starts[j]}, duration {durations[j]}')

