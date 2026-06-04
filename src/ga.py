from dataclasses import dataclass # Just a quick way to cleanup any classes instead of needing an ugly __init__ method
import csv
import json
from multiprocessing import Pool
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

try:
    import geodesic_no_apex
    import fea
    import geodesic
except ImportError:
    import src.fea as fea
    import src.geodesic as geodesic
    import src.geodesic_no_apex as geodesic_no_apex


THICKNESS_MIN = 0.002
THICKNESS_MAX = 0.005
OFFSET_MIN = -0.10
OFFSET_MAX = 0.10
V_CHOICES = (2, 3, 4)

DOME_R = 0.08
DOME_H = 0.08
DOME_VARIANT = "open"  # set to "open" to use the open-top dome variant or "full" for the closed dome

# Physical size limits for the finished dome geometry.
MIN_DOME_RADIUS = 0.07
MAX_DOME_RADIUS = 0.10

SEED = 0

_EXPECTED_LENGTHS_CACHE = {}
_RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

POP_SIZE = 10
GENERATIONS = 10

MUTATION_THICKNESS_SIGMA = 0.10 * (THICKNESS_MAX - THICKNESS_MIN)
MUTATION_OFFSET_SIGMA    = 0.10 * (OFFSET_MAX - OFFSET_MIN)
V_MUTATION_RATE = 0.05


def _dome_backend():
    return geodesic_no_apex if DOME_VARIANT == "open" else geodesic


def _artifact_path(prefix, extension):
    return f"{DOME_VARIANT}_stls/{prefix}_{DOME_VARIANT}_{_RUN_STAMP}.{extension}"


def _ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def best_genome_path():
    return _artifact_path("best_genome", "json")


def log_csv_path():
    return _artifact_path("fitness_history", "csv")


def fitness_plot_path():
    return _artifact_path("fitness_curve", "png")


def best_dome_plot_path():
    return _artifact_path("best_dome", "png")


def _generate_dome(V, radial_offsets=None):
    if DOME_VARIANT == "open":
        return geodesic_no_apex.generate_open_dome(
            R=DOME_R, h=DOME_H, V=V, radial_offsets=radial_offsets,
        )
    return geodesic.generate_dome(
        R=DOME_R, h=DOME_H, V=V, radial_offsets=radial_offsets,
    )


def _visualize_dome(dome, title, path):
    if DOME_VARIANT == "open":
        geodesic_no_apex.visualize_open_dome(dome, title=title, savepath=path)
    else:
        geodesic.visualize_dome(dome, title=title, savepath=path)


def expected_lengths(V):
    key = (DOME_VARIANT, V)
    if key not in _EXPECTED_LENGTHS_CACHE:
        dome = _generate_dome(V)
        n_members = len(dome.members)
        n_orbits = len(geodesic.symmetry_orbits(V))
        _EXPECTED_LENGTHS_CACHE[key] = (n_members, n_orbits)
    return _EXPECTED_LENGTHS_CACHE[key]


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
    dome = _generate_dome(genome.V, radial_offsets=per_vertex_offsets)
    return dome, genome.thicknesses


def _dome_within_radius_limits(dome, min_radius=MIN_DOME_RADIUS, max_radius=MAX_DOME_RADIUS):
    nodes = np.asarray(dome.nodes, dtype=float)
    radii = np.linalg.norm(nodes, axis=1)

    if np.any(radii > max_radius):
        return False

    if np.any(radii < min_radius):
        return False

    if not dome.members:
        return True

    member_nodes = np.asarray(dome.members, dtype=int)
    a = nodes[member_nodes[:, 0]]
    b = nodes[member_nodes[:, 1]]
    ab = b - a
    ab_len_sq = np.einsum("ij,ij->i", ab, ab)

    # Distance from the origin to each strut segment.
    t = -np.einsum("ij,ij->i", a, ab) / np.where(ab_len_sq > 0.0, ab_len_sq, 1.0)
    t = np.clip(t, 0.0, 1.0)
    closest = a + ab * t[:, None]
    closest_radii = np.linalg.norm(closest, axis=1)

    return np.all(closest_radii >= min_radius)
def _evaluate_uncached(genome):
    dome, thicknesses = decode(genome)
    if not _dome_within_radius_limits(dome):
        return 0.0
    try:
        model = fea.analyze_structure(dome, thicknesses)
    except Exception:
        return 0.0
    # Fitness is the dome's failure load divided by its total mass.
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
def save_genome(genome, path):
    _ensure_parent_dir(path)
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
    _ensure_parent_dir(path)
    history = np.asarray(history)
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(history[:, 0], label="best", color="green")
    ax.plot(history[:, 1], label="mean", color="steelblue")
    ax.plot(history[:, 2], label="worst", color="red", alpha=0.5)
    ax.set_xlabel("generation")
    ax.set_ylabel("strength-to-weight ratio (N/kg)")
    ax.set_title("GA fitness over generations")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")

def draw_history(history, ax):
    history = np.asarray(history)
    ax.clear()
    ax.set_title("Dome Optimization Progress")
    if len(history) > 0:
        ax.plot(history[:, 0], label="best", color="green")
        ax.plot(history[:, 1], label="mean", color="steelblue")
        ax.plot(history[:, 2], label="worst", color="red", alpha=0.5)
        ax.legend()
    ax.set_xlabel("generation")
    ax.set_ylabel("strength-to-weight ratio (N/kg)")
    ax.grid(alpha=0.3)
    return ax.figure


def create_gui_graph(history):
    fig, ax = plt.subplots()
    draw_history(history, ax)
    return fig


def visualize_genome(genome, path, title=None):
    dome, thicknesses = decode(genome)
    if title is None:
        mass = fea.total_mass(dome, thicknesses)
        title = (f"Best {DOME_VARIANT} dome  V={genome.V}  members={len(dome.members)}  "
                 f"mass={mass:.1f} kg")
    _visualize_dome(dome, title, path)


def set_params(radius, height, min_thick, max_thick, min_offset, max_offset, seed=0):
    global DOME_R, DOME_H, THICKNESS_MIN, THICKNESS_MAX, OFFSET_MIN, OFFSET_MAX, SEED
    DOME_R = radius
    DOME_H = height
    THICKNESS_MIN = min_thick
    THICKNESS_MAX = max_thick
    OFFSET_MIN = min_offset
    OFFSET_MAX = max_offset
    SEED = seed


def run_ga(progress_callback=None):
    rng = np.random.default_rng(seed=SEED)

    population = [random_genome(V=2, rng=rng) for _ in range(POP_SIZE)]
    with Pool() as pool:
        fitness = pool.map(_evaluate_uncached, population)

    history = []
    best_ever = -float("inf")

    _ensure_parent_dir(log_csv_path())
    with open(log_csv_path(), "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["generation", "best", "mean", "worst"])

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
            with Pool() as pool:
                fitness = pool.map(_evaluate_uncached, population)

            best = max(fitness)
            mean = sum(fitness) / len(fitness)
            worst = min(fitness)

            history.append((best, mean, worst))
            writer.writerow([gen, best, mean, worst])
            csv_file.flush()

            if best > best_ever:
                best_ever = best
                best_idx = fitness.index(best)
                save_genome(population[best_idx], best_genome_path())

            print(f"Gen {gen:3d}  best={best:8.2f}  mean={mean:8.2f}  "
                                    f"worst={worst:8.2f}")

            if progress_callback is not None:
                progress_callback(list(history))

    plot_fitness(history, fitness_plot_path())
    visualize_genome(
        load_genome(best_genome_path()),
        best_dome_plot_path(),
        title=f"Best {DOME_VARIANT} dome  strength-to-weight={best_ever:.2f} N/kg",
    )
    print(f"\nDone. All-time best strength-to-weight ratio: {best_ever:.2f}")
    print(f"Saved: {log_csv_path()}, {best_genome_path()}, "
          f"{fitness_plot_path()}, {best_dome_plot_path()}")

if __name__ == "__main__":
    run_ga()