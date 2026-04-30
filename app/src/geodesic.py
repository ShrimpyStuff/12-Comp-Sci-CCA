"""Icosahedral Class-I geodesic dome generator.

Pipeline:
  1. Build unit icosahedron with a vertex at the north pole.
  2. Class-I subdivide each face into V*V sub-triangles; project each new
     vertex to the unit sphere; dedupe across faces.
  3. Scale/translate so the apex sits at (0, 0, h) and the z=0 slice of the
     sphere is a circle of radius R.
  4. Truncate at z=0 by triangle clipping. Edges that cross z=0 produce new
     base-ring nodes, which are projected horizontally onto the circle r=R
     so all base nodes lie exactly on the ground circle.

Convention used (documented per spec):
  - Class-I subdivision: each edge of each icosahedron face is split into V
    equal segments; sub-triangles on the face correspond to the (i,j,k)
    barycentric lattice with i+j+k=V. New vertices are then projected to the
    sphere from the sphere center (origin on the unit sphere).
"""
from collections import namedtuple
import numpy as np

# from app.src import fea
import fea



Dome = namedtuple("Dome", "nodes members base_ids apex_id")


def _icosahedron():
    """Unit icosahedron, vertex 0 at the north pole (0,0,1).

    Returns (verts (12,3), faces (20,3)). Faces are CCW viewed from outside.
    """
    z = 1.0 / np.sqrt(5.0)
    r = 2.0 / np.sqrt(5.0)
    v = [(0.0, 0.0, 1.0)]
    for k in range(5):
        t = 2.0 * np.pi * k / 5.0
        v.append((r * np.cos(t), r * np.sin(t), z))
    for k in range(5):
        t = 2.0 * np.pi * (k + 0.5) / 5.0
        v.append((r * np.cos(t), r * np.sin(t), -z))
    v.append((0.0, 0.0, -1.0))
    verts = np.array(v)

    faces = []
    # top cap
    for k in range(5):
        faces.append((0, 1 + k, 1 + (k + 1) % 5))
    # bottom cap
    for k in range(5):
        faces.append((11, 6 + (k + 1) % 5, 6 + k))
    # middle strip
    for k in range(5):
        u = 1 + k
        u_next = 1 + (k + 1) % 5
        l_prev = 6 + (k - 1) % 5
        l = 6 + k
        faces.append((u, l_prev, l))       # apex-up
        faces.append((l, u_next, u))       # apex-down
    return verts, np.array(faces, dtype=int)


def _subdivide_class1(verts, faces, V):
    """Subdivide each triangle into V*V sub-triangles and project to unit sphere.

    Shared vertices across adjacent faces are deduplicated via coordinate
    quantization at 1e-10 (safe for this geometry).
    """
    out_verts = []
    cache = {}

    def get_idx(p):
        p = p / np.linalg.norm(p)
        key = tuple(np.round(p, 10))
        idx = cache.get(key)
        if idx is None:
            idx = len(out_verts)
            out_verts.append(p)
            cache[key] = idx
        return idx

    out_faces = []
    for fa, fb, fc in faces:
        A, B, C = verts[fa], verts[fb], verts[fc]
        grid = {}
        for i in range(V + 1):
            for j in range(V + 1 - i):
                k = V - i - j
                p = (k * A + i * B + j * C) / V
                grid[(i, j)] = get_idx(p)
        for i in range(V):
            for j in range(V - i):
                out_faces.append((grid[(i, j)], grid[(i + 1, j)], grid[(i, j + 1)]))
        for i in range(V - 1):
            for j in range(V - 1 - i):
                out_faces.append((grid[(i + 1, j)], grid[(i + 1, j + 1)], grid[(i, j + 1)]))

    return np.array(out_verts), np.array(out_faces, dtype=int)


def generate_dome(R, h, V, radial_offsets=None):
    """Generate a geodesic dome truncated at z=0.

    Parameters
    ----------
    R : float
        Ground base radius (m). Base nodes lie on the circle x^2+y^2=R^2, z=0.
    h : float
        Dome height (m); apex lies at (0, 0, h) when radial_offsets is None
        (or when the apex offset is zero).
    V : int
        Subdivision frequency (>=1).
    radial_offsets : array-like or None
        Optional per-vertex radial offsets applied to unit-sphere vertices
        before scaling and truncation. Each vertex is scaled by (1 + offset),
        i.e. moved along its own radial direction. Length must match the
        number of subdivided unit-sphere vertices for this V. The caller
        is responsible for keeping the apex (vertex 0) offset at zero if
        the apex must remain at (0, 0, h).

    Returns
    -------
    Dome(nodes, members, base_ids, apex_id)
    """
    assert V >= 1
    assert R > 0 and h > 0

    v_unit, f = _icosahedron()
    v_unit, f = _subdivide_class1(v_unit, f, V)

    if radial_offsets is not None:
        radial_offsets = np.asarray(radial_offsets, dtype=float)
        assert radial_offsets.shape == (len(v_unit),), (
            f"need {len(v_unit)} radial offsets, got {radial_offsets.shape}"
        )
        # Each unit-sphere vertex is its own radial unit vector; scale by (1 + delta)
        v_unit = v_unit * (1.0 + radial_offsets)[:, None]

    # Scale so z=0 circle of the scaled sphere has radius R and apex at (0,0,h).
    R_s = (R * R + h * h) / (2.0 * h)
    z_c = (h * h - R * R) / (2.0 * h)
    v = R_s * v_unit + np.array([0.0, 0.0, z_c])

    EPS_Z = 1e-9
    node_list = []
    orig_map = {}   # original index -> new index
    clip_map = {}   # frozenset({i,j}) -> new index (clipped at z=0)
    edges = set()

    def add_orig(i):
        nid = orig_map.get(i)
        if nid is None:
            nid = len(node_list)
            node_list.append(v[i].copy())
            orig_map[i] = nid
        return nid

    def add_clip(i, j):
        # i is above z=0, j is below. If i is on z=0, reuse the original node.
        if abs(v[i, 2]) < EPS_Z:
            return add_orig(i)
        key = frozenset({i, j})
        nid = clip_map.get(key)
        if nid is not None:
            return nid
        zi, zj = v[i, 2], v[j, 2]
        t = zi / (zi - zj)
        p = v[i] + t * (v[j] - v[i])
        r_xy = np.hypot(p[0], p[1])
        if r_xy > 1e-12:
            p[0] *= R / r_xy
            p[1] *= R / r_xy
        p[2] = 0.0
        nid = len(node_list)
        node_list.append(p)
        clip_map[key] = nid
        return nid

    def add_edge(a, b):
        if a != b:
            edges.add((min(a, b), max(a, b)))

    for fa, fb, fc in f:
        tri = (fa, fb, fc)
        z_vals = v[list(tri), 2]
        keep = z_vals > -EPS_Z
        n_keep = int(keep.sum())
        if n_keep == 0:
            continue
        if n_keep == 3:
            n = [add_orig(x) for x in tri]
            add_edge(n[0], n[1]); add_edge(n[1], n[2]); add_edge(n[2], n[0])
            continue
        above = [x for x, k in zip(tri, keep) if k]
        below = [x for x, k in zip(tri, keep) if not k]
        if n_keep == 2:
            a, b = above
            c = below[0]
            na, nb = add_orig(a), add_orig(b)
            nac, nbc = add_clip(a, c), add_clip(b, c)
            add_edge(na, nb)
            add_edge(na, nac)
            add_edge(nb, nbc)
            add_edge(nac, nbc)
        else:  # n_keep == 1
            a = above[0]
            b, c = below
            na = add_orig(a)
            nab, nac = add_clip(a, b), add_clip(a, c)
            add_edge(na, nab)
            add_edge(na, nac)
            add_edge(nab, nac)

    nodes = np.array(node_list)
    base_ids = np.where(np.abs(nodes[:, 2]) < 1e-6)[0].tolist()
    apex_id = int(np.argmax(nodes[:, 2]))
    members = sorted(edges)
    return Dome(nodes=nodes, members=members, base_ids=base_ids, apex_id=apex_id)


def symmetry_orbits(V, tol=6):
    """Group unit-sphere vertices into 5-fold rotational orbits about z.

    Returns a list of orbits; each orbit is a list of vertex indices that are
    rotational equivalents (same cylindrical r and z within tol decimal places).
    The vertex order matches the array generate_dome consumes via radial_offsets.
    """
    v_unit, _f = _icosahedron()
    v_unit, _f = _subdivide_class1(v_unit, _f, V)
    groups = {}
    for i, p in enumerate(v_unit):
        r = float(np.hypot(p[0], p[1]))
        z = float(p[2])
        key = (round(r, tol), round(z, tol))
        groups.setdefault(key, []).append(i)
    return list(groups.values())


def visualize_dome(dome, title="Geodesic dome", savepath=None, ax=None):
    """Render the dome as a wireframe with apex (red) and base ring (green)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    fig = None
    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")

    nodes = dome.nodes
    segs = [(nodes[i], nodes[j]) for (i, j) in dome.members]
    lc = Line3DCollection(segs, colors="steelblue", linewidths=0.9, alpha=0.9)
    ax.add_collection3d(lc)
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c="k", s=6)
    ax.scatter(*nodes[dome.apex_id], c="red", s=60, label=f"apex")
    if dome.base_ids:
        bn = nodes[dome.base_ids]
        ax.scatter(bn[:, 0], bn[:, 1], bn[:, 2], c="green", s=20,
                   label=f"base ring ({len(dome.base_ids)} nodes)")

    lims = np.array([nodes.min(0), nodes.max(0)]).T
    center = lims.mean(axis=1)
    size = float((lims[:, 1] - lims[:, 0]).max())
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(c - size / 2.0, c + size / 2.0)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_title(title)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.legend(loc="upper right", fontsize=8)
    if savepath:
        plt.savefig(savepath, dpi=120, bbox_inches="tight")
    return fig, ax


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R, h = 5.0, 5.0  # hemisphere preview
    rows = []
    V=2
    d = generate_dome(R, h, V)
    rows.append((V, len(d.nodes), len(d.members), len(d.base_ids),
                    float(d.nodes[d.apex_id, 2])))
    visualize_dome(d, title=f"Geodesic dome V={V} (R={R} m, h={h} m)",
                    savepath=f"dome_V{V}.png")
    thicknesses = np.full(len(d.members), 0.01)  # uniform 1cm rod radius for the demo
    output = fea.analyze_structure(d, thicknesses)
    plt.close("all")

    print(f"{'V':>3} {'nodes':>7} {'members':>9} {'base':>6} {'apex_z':>8}")
    for V, n, m, b, z in rows:
        print(f"{V:>3d} {n:>7d} {m:>9d} {b:>6d} {z:>8.3f}")
