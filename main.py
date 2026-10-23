
import random
import matplotlib.pyplot as plt

random.seed(7)

SIZE = 20
STARTS = [(1, 1), (1, 18), (18, 1)]
GOALS = [(18, 18), (18, 1), (1, 1)]
TARGETS = [(5, 15), (10, 10), (15, 15), (15, 5), (7, 7)]
HAZARDS = {(x, y) for x in range(3, 7) for y in range(3, 7)}
NO_FLY = {(8, y) for y in range(5, 14)}

# A chromosome contains 2 waypoints for each of 3 UAVs
def create_chromosome():
    return [[(random.randint(1, 18), random.randint(1, 18))
             for _ in range(2)] for _ in range(3)]

def make_path(start, genes, goal):
    path = [start]
    for tx, ty in genes + [goal]:
        x, y = path[-1]
        while x != tx:
            x += 1 if tx > x else -1
            path.append((x, y))
        while y != ty:
            y += 1 if ty > y else -1
            path.append((x, y))
    return path

class GeneticAlgorithm:
    def __init__(self, population=30, generations=40, mutation=0.15):
        self.size = population
        self.generations = generations
        self.mutation = mutation

    def paths(self, chromosome):
        return [make_path(STARTS[i], chromosome[i], GOALS[i])
                for i in range(3)]

    def fitness(self, chromosome):
        paths = self.paths(chromosome)
        cells = {p for path in paths for p in path}
        coverage = len(cells & set(TARGETS))
        distance = sum(len(p) - 1 for p in paths)
        risk = sum(p in HAZARDS for path in paths for p in path)
        violations = sum(p in NO_FLY for path in paths for p in path)
        return 3 * coverage - 0.05 * distance - 0.5 * risk - 5 * violations

    def run(self):
        pop = [create_chromosome() for _ in range(self.size)]
        best_history, avg_history = [], []

        for _ in range(self.generations):
            pop.sort(key=self.fitness, reverse=True)
            scores = [self.fitness(c) for c in pop]
            best_history.append(scores[0])
            avg_history.append(sum(scores) / len(scores))

            new_pop = [pop[0]]  # Keep the best solution
            while len(new_pop) < self.size:
                p1 = max(random.sample(pop, 3), key=self.fitness)
                p2 = max(random.sample(pop, 3), key=self.fitness)
                cut = random.randint(1, 2)
                child = [p1[i][:] if i < cut else p2[i][:]
                         for i in range(3)]

                for route in child:
                    for j, (x, y) in enumerate(route):
                        if random.random() < self.mutation:
                            route[j] = (
                                max(1, min(18, x + random.randint(-3, 3))),
                                max(1, min(18, y + random.randint(-3, 3)))
                            )
                new_pop.append(child)
            pop = new_pop

        best = max(pop, key=self.fitness)
        return best, best_history, avg_history

# Run Genetic Algorithm
ga = GeneticAlgorithm()
best, best_fit, avg_fit = ga.run()
paths = ga.paths(best)

print("Best GA fitness:", round(ga.fitness(best), 2))
print("UAV routes:", best)
print("Covered targets:", len(
    {p for path in paths for p in path} & set(TARGETS)
))

# Plot 1: Fitness convergence
plt.figure(figsize=(7, 4))
plt.plot(best_fit, label="Best fitness")
plt.plot(avg_fit, label="Average fitness")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("GA Fitness Convergence")
plt.legend()
plt.grid()
plt.show()

# Plot 2: Routes and coverage map
plt.figure(figsize=(7, 6))
for x, y in HAZARDS:
    plt.scatter(x, y, color="red", s=25)
for x, y in NO_FLY:
    plt.scatter(x, y, color="black", s=25)
for x, y in TARGETS:
    plt.scatter(x, y, color="gold", marker="*", s=150)

colors = ["blue", "green", "purple"]
for i, path in enumerate(paths):
    plt.plot([p[0] for p in path], [p[1] for p in path],
             color=colors[i], label=f"UAV {i+1}")
    plt.scatter(*STARTS[i], color=colors[i], marker="o")
    plt.scatter(*GOALS[i], color=colors[i], marker="X")

plt.xlim(0, 19)
plt.ylim(0, 19)
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.title("UAV Routes, Targets and Hazard Grid")
plt.legend()
plt.grid()
plt.show()

# Baseline: Random search using the same fitness function
random_best = max((create_chromosome() for _ in range(1200)),
                  key=ga.fitness)

print("Random search fitness:", round(ga.fitness(random_best), 2))
print("GA fitness:", round(ga.fitness(best), 2))
