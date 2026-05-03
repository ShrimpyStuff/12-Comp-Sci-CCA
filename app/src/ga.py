from dataclasses import dataclass
import csv
import json

import numpy as np
import matplotlib.pyplot as plt
import random

import fea
import geodesic


THICKNESS_MIN = 0.005
THICKNESS_MAX = 0.05
OFFSET_MIN = -0.10
OFFSET_MAX = 0.10
V_CHOICES = (2, 3, 4)

DOME_R = 5.0
DOME_H = 5.0


_EXPECTED_LENGTHS_CACHE = {}

POP_SIZE = 60
GENERATIONS = 100

MUTATION_THICKNESS_SIGMA = 0.10 * (THICKNESS_MAX - THICKNESS_MIN)
MUTATION_OFFSET_SIGMA    = 0.10 * (OFFSET_MAX - OFFSET_MIN)
V_MUTATION_RATE = 0.05


def expected_lengths(V):
    if V not in _EXPECTED_LENGTHS_CACHE:
        dome = geodesic.generate_dome(R=DOME_R, h=DOME_H, V=V)
        n_members = len(dome.members)
        n_orbits = len(geodesic.symmetry_orbits(V))
        _EXPECTED_LENGTHS_CACHE[V] = (n_members, n_orbits)
    return _EXPECTED_LENGTHS_CACHE[V]


@dataclass
class Genome:
    V: int
    thicknesses: np.ndarray
    offsets: np.ndarray

    def __post_init__(self):
        self.thicknesses = np.asarray(self.thicknesses, dtype=float)
        self.offsets = np.asarray(self.offsets, dtype=float)

        assert self.V in V_CHOICES, f"V={self.V} not in {V_CHOICES}"

        n_members, n_orbits = expected_lengths(self.V)
        assert self.thicknesses.shape == (n_members,), (
            f"thicknesses needs length {n_members}, got {self.thicknesses.shape}"
        )
        assert self.offsets.shape == (n_orbits,), (
            f"offsets needs length {n_orbits}, got {self.offsets.shape}"
        )


def random_genome(V=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if V is None:
        V = int(rng.choice(V_CHOICES))
    n_members, n_orbits = expected_lengths(V)
    thicknesses = rng.uniform(THICKNESS_MIN, THICKNESS_MAX, size=n_members)
    offsets = rng.uniform(OFFSET_MIN, OFFSET_MAX, size=n_orbits)
    return Genome(V=V, thicknesses=thicknesses, offsets=offsets)


def expand_offsets(V, per_orbit_offsets):
    orbits = geodesic.symmetry_orbits(V)
    n_verts = sum(len(o) for o in orbits)
    per_vertex = np.zeros(n_verts, dtype=float)
    for orbit, offset in zip(orbits, per_orbit_offsets):
        for v in orbit:
            per_vertex[v] = offset
    per_vertex[0] = 0.0
    return per_vertex


def decode(genome):
    per_vertex_offsets = expand_offsets(genome.V, genome.offsets)
    dome = geodesic.generate_dome(
        R=DOME_R, h=DOME_H, V=genome.V,
        radial_offsets=per_vertex_offsets,
    )
    return dome, genome.thicknesses


_FITNESS_CACHE = {}
_CACHE_HITS = 0
_CACHE_MISSES = 0


def _genome_key(genome):
    return (genome.V, genome.thicknesses.tobytes(), genome.offsets.tobytes())


def cached_fitness(genome, compute):
    global _CACHE_HITS, _CACHE_MISSES
    key = _genome_key(genome)
    if key in _FITNESS_CACHE:
        _CACHE_HITS += 1
        return _FITNESS_CACHE[key]
    _CACHE_MISSES += 1
    value = compute()
    _FITNESS_CACHE[key] = value
    return value


def cache_stats():
    return _CACHE_HITS, _CACHE_MISSES, len(_FITNESS_CACHE)


def evaluate(genome):
    return cached_fitness(genome, lambda: _evaluate_uncached(genome))


def _evaluate_uncached(genome):
    dome, thicknesses = decode(genome)
    try:
        model = fea.analyze_structure(dome, thicknesses)
    except Exception:
        return 0.0
    return fea.specific_strength(model, dome, thicknesses)


def tournament_selection(population, fitness, k=3, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    fitness = np.asarray(fitness)
    idx = rng.choice(len(population), size=k, replace=False)
    winner = idx[np.argmax(fitness[idx])]
    return population[winner]


CROSSOVER_RATE = 0.8


def crossover(p1, p2, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    assert p1.V == p2.V, f"crossover needs matching V, got {p1.V} vs {p2.V}"

    t_mask = rng.random(p1.thicknesses.shape) < 0.5
    o_mask = rng.random(p1.offsets.shape) < 0.5

    child_thicknesses = np.where(t_mask, p1.thicknesses, p2.thicknesses)
    child_offsets = np.where(o_mask, p1.offsets, p2.offsets)

    return Genome(V=p1.V, thicknesses=child_thicknesses, offsets=child_offsets)


def clone(genome):
    return Genome(
        V=genome.V,
        thicknesses=genome.thicknesses.copy(),
        offsets=genome.offsets.copy(),
    )


def mutate(genome, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if rng.random() < V_MUTATION_RATE:
        i = V_CHOICES.index(genome.V)
        delta = int(rng.choice([-1, 1]))
        new_i = max(0, min(len(V_CHOICES) - 1, i + delta))
        new_V = V_CHOICES[new_i]
        if new_V != genome.V:
            return random_genome(V=new_V, rng=rng)

    thicknesses = genome.thicknesses.copy()
    offsets = genome.offsets.copy()
    N = len(thicknesses) + len(offsets)
    p = 1.0 / N

    t_mask = rng.random(thicknesses.shape) < p
    o_mask = rng.random(offsets.shape) < p

    t_noise = rng.normal(0.0, MUTATION_THICKNESS_SIGMA, thicknesses.shape)
    o_noise = rng.normal(0.0, MUTATION_OFFSET_SIGMA, offsets.shape)

    thicknesses = np.where(t_mask, thicknesses + t_noise, thicknesses)
    offsets     = np.where(o_mask, offsets + o_noise, offsets)

    thicknesses = np.clip(thicknesses, THICKNESS_MIN, THICKNESS_MAX)
    offsets     = np.clip(offsets, OFFSET_MIN, OFFSET_MAX)

    return Genome(V=genome.V, thicknesses=thicknesses, offsets=offsets)


LOG_CSV_PATH = "fitness_history.csv"
BEST_GENOME_PATH = "best_genome.json"
FITNESS_PLOT_PATH = "fitness_curve.png"
BEST_DOME_PLOT_PATH = "best_dome.png"


def save_genome(genome, path):
    data = {
        "V": int(genome.V),
        "thicknesses": genome.thicknesses.tolist(),
        "offsets": genome.offsets.tolist(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_genome(path):
    with open(path) as f:
        data = json.load(f)
    return Genome(
        V=data["V"],
        thicknesses=np.array(data["thicknesses"]),
        offsets=np.array(data["offsets"]),
    )


def plot_fitness(history, path):
    history = np.asarray(history)
    plt.figure(figsize=(8, 5))
    plt.plot(history[:, 0], label="best",  color="green")
    plt.plot(history[:, 1], label="mean",  color="steelblue")
    plt.plot(history[:, 2], label="worst", color="red", alpha=0.5)
    plt.xlabel("generation")
    plt.ylabel("specific strength (N/kg)")
    plt.title("GA fitness over generations")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def visualize_genome(genome, path, title=None):
    dome, thicknesses = decode(genome)
    if title is None:
        mass = fea.total_mass(dome, thicknesses)
        title = (f"Best dome  V={genome.V}  members={len(dome.members)}  "
                 f"mass={mass:.1f} kg")
    geodesic.visualize_dome(dome, title=title, savepath=path)
    plt.close("all")


if __name__ == "__main__":
    rng = np.random.default_rng(seed=0)

    population = [random_genome(V=2, rng=rng) for _ in range(POP_SIZE)]
    fitness = [evaluate(g) for g in population]

    history = []
    best_ever = -float("inf")

    with open(LOG_CSV_PATH, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["generation", "best", "mean", "worst",
                         "cache_hits", "cache_misses"])

        for gen in range(GENERATIONS):
            next_pop = []
            while len(next_pop) < POP_SIZE:
                p1 = tournament_selection(population, fitness, k=3, rng=rng)
                p2 = tournament_selection(population, fitness, k=3, rng=rng)
                if p1.V == p2.V and rng.random() < CROSSOVER_RATE:
                    child = crossover(p1, p2, rng=rng)
                else:
                    child = clone(p1)
                child = mutate(child, rng=rng)
                next_pop.append(child)
            population = next_pop
            fitness = [evaluate(g) for g in population]

            best = max(fitness)
            mean = sum(fitness) / len(fitness)
            worst = min(fitness)
            hits, misses, _ = cache_stats()

            history.append((best, mean, worst))
            writer.writerow([gen, best, mean, worst, hits, misses])
            csv_file.flush()

            if best > best_ever:
                best_ever = best
                best_idx = fitness.index(best)
                save_genome(population[best_idx], BEST_GENOME_PATH)

            print(f"Gen {gen:3d}  best={best:8.2f}  mean={mean:8.2f}  "
                  f"worst={worst:8.2f}  cache={hits}/{hits+misses}")

    plot_fitness(history, FITNESS_PLOT_PATH)
    visualize_genome(load_genome(BEST_GENOME_PATH), BEST_DOME_PLOT_PATH,
                     title=f"Best dome  fitness={best_ever:.2f} N/kg")
    print(f"\nDone. All-time best fitness: {best_ever:.2f}")
    print(f"Saved: {LOG_CSV_PATH}, {BEST_GENOME_PATH}, "
          f"{FITNESS_PLOT_PATH}, {BEST_DOME_PLOT_PATH}")
