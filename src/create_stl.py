from pathlib import Path
import json

import numpy as np
import trimesh

try:
    from . import geodesic_no_apex
    from . import geodesic
except ImportError:
    import geodesic_no_apex
    import geodesic

DEFAULT_DOME_R = 5.0
DEFAULT_DOME_H = 5.0
DOME_VARIANT = "open"  # set to "normal" for the standard apex dome
NODE_SPHERE_SCALE = 1.0
NODE_SPHERE_SECTIONS = 2


def _variant_suffix():
    return "open" if DOME_VARIANT == "open" else "normal"


def _genome_path():
    filename = "best_genome_open.json" if DOME_VARIANT == "open" else "best_genome.json"
    return Path(__file__).resolve().parents[2] / filename


def _expand_offsets(V, per_orbit_offsets):
    if per_orbit_offsets is None:
        return None

    orbits = geodesic.symmetry_orbits(V)
    n_verts = sum(len(orbit) for orbit in orbits)
    per_vertex = np.zeros(n_verts, dtype=float)
    for orbit, offset in zip(orbits, per_orbit_offsets):
        for vertex in orbit:
            per_vertex[vertex] = offset
    per_vertex[0] = 0.0
    return per_vertex


def _rotation_from_z(direction):
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)

    z_axis = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z_axis, direction)
    sine = np.linalg.norm(cross)
    cosine = float(np.dot(z_axis, direction))

    if sine < 1e-12:
        if cosine > 0.0:
            rotation = np.eye(3)
        else:
            rotation = np.array([
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ])
    else:
        vx = np.array([
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ])
        rotation = np.eye(3) + vx + vx @ vx * ((1.0 - cosine) / (sine * sine))

    transform = np.eye(4)
    transform[:3, :3] = rotation
    return transform


def _cylinder_between(start, end, radius, sections=32):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    vector = end - start
    height = float(np.linalg.norm(vector))
    if height <= 0.0:
        return None

    mesh = trimesh.creation.cylinder(radius=float(radius), height=height, sections=sections)
    transform = _rotation_from_z(vector)
    transform[:3, 3] = (start + end) / 2.0
    mesh.apply_transform(transform)
    return mesh


def _sphere_at(center, radius, sections=NODE_SPHERE_SECTIONS):
    if radius <= 0.0:
        return None

    mesh = trimesh.creation.icosphere(subdivisions=sections, radius=float(radius))
    mesh.apply_translation(np.asarray(center, dtype=float))
    return mesh


def _normalize_thicknesses(thicknesses, n_members):
    if np.isscalar(thicknesses):
        radius = np.asarray(thicknesses, dtype=float).item()
        return np.full(n_members, radius, dtype=float)

    thicknesses = np.asarray(thicknesses, dtype=float)
    if thicknesses.shape != (n_members,):
        raise ValueError(f"need {n_members} thickness values, got {thicknesses.shape}")
    return thicknesses


def _load_best_genome(genome_path):
    genome_path = Path(genome_path)
    with genome_path.open() as f:
        data = json.load(f)
    return data


def _generate_dome(R, h, V, radial_offsets=None):
    if DOME_VARIANT == "open":
        return geodesic_no_apex.generate_open_dome(R=R, h=h, V=V, radial_offsets=radial_offsets)
    return geodesic.generate_dome(R=R, h=h, V=V, radial_offsets=radial_offsets)


def create_stl(filename, genome_path=None, R=DEFAULT_DOME_R, h=DEFAULT_DOME_H,
               V=None, thicknesses=None, offsets=None):
    if genome_path is None:
        genome_path = _genome_path()
    genome = _load_best_genome(genome_path)
    if V is None:
        V = int(genome["V"])
    if thicknesses is None:
        thicknesses = genome["thicknesses"]
    if offsets is None:
        offsets = genome["offsets"]

    per_vertex_offsets = _expand_offsets(V, offsets)
    dome = _generate_dome(R=R, h=h, V=V, radial_offsets=per_vertex_offsets)
    radii = _normalize_thicknesses(thicknesses, len(dome.members))

    cylinders = []
    for (n1, n2), radius in zip(dome.members, radii):
        mesh = _cylinder_between(dome.nodes[n1], dome.nodes[n2], radius)
        if mesh is not None:
            cylinders.append(mesh)

    node_radii = np.zeros(len(dome.nodes), dtype=float)
    for (n1, n2), radius in zip(dome.members, radii):
        node_radii[n1] = max(node_radii[n1], radius)
        node_radii[n2] = max(node_radii[n2], radius)

    for node_index, radius in enumerate(node_radii):
        mesh = _sphere_at(dome.nodes[node_index], radius * NODE_SPHERE_SCALE)
        if mesh is not None:
            cylinders.append(mesh)

    mesh = trimesh.util.concatenate(cylinders) if cylinders else trimesh.Trimesh()

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)
    return mesh
    
if __name__ == "__main__":
    create_stl(f"best_dome_{_variant_suffix()}.stl")