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

try:
    import fea
except ImportError:
    import src.fea as fea



Dome = namedtuple("Dome", "nodes members base_ids apex_id")


def _icosahedron():
    """Unit icosahedron, vertex 0 at the north pole (0,0,1).

    Returns (verts (12,3), faces (20,3)). Faces are CCW viewed from outside.
    """
    # An icosahedron has 12 vertices. Placed on the unit sphere they form:
    # the north pole, a ring of 5 at height +z, a ring of 5 at height -z
    # (rotated half a step so the rings interlock), and the south pole.
    z = 1.0 / np.sqrt(5.0)
    r = 2.0 / np.sqrt(5.0)          # horizontal radius of each ring
    v = [(0.0, 0.0, 1.0)]          # vertex 0: north pole
    for k in range(5):
        t = 2.0 * np.pi * k / 5.0  # evenly spaced angle around the ring
        v.append((r * np.cos(t), r * np.sin(t), z))
    for k in range(5):
        t = 2.0 * np.pi * (k + 0.5) / 5.0  # +0.5 step offsets the lower ring
        v.append((r * np.cos(t), r * np.sin(t), -z))
    v.append((0.0, 0.0, -1.0))    # vertex 11: south pole
    verts = np.array(v)

    # Each face is a triangle given as 3 vertex indices, wound CCW from outside.
    faces = []
    # top cap: 5 triangles fanning out from the north pole
    for k in range(5):
        faces.append((0, 1 + k, 1 + (k + 1) % 5))
    # bottom cap: 5 triangles fanning out from the south pole
    for k in range(5):
        faces.append((11, 6 + (k + 1) % 5, 6 + k))
    # middle strip: 10 triangles linking the two rings into a zig-zag band
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
        # Push the point onto the unit sphere, then look it up by its rounded
        # coordinates so a vertex shared by two faces is only stored once.
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
        # Build a triangular lattice of points across the face. (i, j) are
        # barycentric-style counts: p is a weighted blend of corners A, B, C,
        # so (i, j) steps move us in a grid over the triangle.
        grid = {}
        for i in range(V + 1):
            for j in range(V + 1 - i):
                k = V - i - j
                p = (k * A + i * B + j * C) / V
                grid[(i, j)] = get_idx(p)
        # Tile the lattice with small triangles. Two passes are needed because
        # a subdivided triangle is filled by upward- and downward-pointing ones.
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

    # We want a spherical cap: apex at height h, and the slice at z=0 a circle
    # of radius R. Solving those two conditions gives the sphere radius R_s and
    # how far its centre sits below the apex (z_c). The unit sphere is then
    # scaled by R_s and shifted up by z_c to match.
    R_s = (R * R + h * h) / (2.0 * h)
    z_c = (h * h - R * R) / (2.0 * h)
    v = R_s * v_unit + np.array([0.0, 0.0, z_c])

    # EPS_Z is a small tolerance so points sitting essentially on z=0 are
    # treated as exactly on the ground, not slightly above or below.
    EPS_Z = 1e-9
    node_list = []
    orig_map = {}   # original index -> new index (only nodes we actually keep)
    clip_map = {}   # frozenset({i,j}) -> new index for a node created on z=0
    edges = set()

    def add_orig(i):
        # Add a kept vertex once, reusing its new index if seen before.
        nid = orig_map.get(i)
        if nid is None:
            nid = len(node_list)
            node_list.append(v[i].copy())
            orig_map[i] = nid
        return nid

    def add_clip(i, j):
        # Edge i->j crosses the ground: i is above z=0, j is below. Create the
        # crossing point. If i is already on z=0, just reuse it.
        if abs(v[i, 2]) < EPS_Z:
            return add_orig(i)
        key = frozenset({i, j})
        nid = clip_map.get(key)
        if nid is not None:
            return nid
        # Linearly interpolate along the edge to find where z hits 0.
        zi, zj = v[i, 2], v[j, 2]
        t = zi / (zi - zj)
        p = v[i] + t * (v[j] - v[i])
        # Push the new point straight out to radius R so every base node lands
        # exactly on the ground circle.
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
        # Store an undirected edge with a sorted key so duplicates collapse.
        if a != b:
            edges.add((min(a, b), max(a, b)))

    # Walk every triangle and keep only the part at or above the ground (z=0).
    for fa, fb, fc in f:
        tri = (fa, fb, fc)
        z_vals = v[list(tri), 2]
        keep = z_vals > -EPS_Z          # which corners are above the ground
        n_keep = int(keep.sum())
        if n_keep == 0:
            continue                    # whole triangle is below ground: drop it
        if n_keep == 3:
            # Entirely above ground: keep all three edges as-is.
            n = [add_orig(x) for x in tri]
            add_edge(n[0], n[1]); add_edge(n[1], n[2]); add_edge(n[2], n[0])
            continue
        # Otherwise the triangle straddles z=0 and must be clipped.
        above = [x for x, k in zip(tri, keep) if k]
        below = [x for x, k in zip(tri, keep) if not k]
        if n_keep == 2:
            # Two corners above, one below: the kept piece is a quad (the two
            # top corners plus two new ground points where the edges cross).
            a, b = above
            c = below[0]
            na, nb = add_orig(a), add_orig(b)
            nac, nbc = add_clip(a, c), add_clip(b, c)
            add_edge(na, nb)
            add_edge(na, nac)
            add_edge(nb, nbc)
            add_edge(nac, nbc)
        else:  # n_keep == 1
            # One corner above, two below: the kept piece is a small triangle
            # (the top corner plus two new ground points).
            a = above[0]
            b, c = below
            na = add_orig(a)
            nab, nac = add_clip(a, b), add_clip(a, c)
            add_edge(na, nab)
            add_edge(na, nac)
            add_edge(nab, nac)

    nodes = np.array(node_list)
    # Base nodes are everything sitting on the ground; the apex is the highest.
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
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    fig: Figure
    if ax is None:
        fig = Figure(figsize=(7, 7))
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
        fig.savefig(savepath, dpi=120, bbox_inches="tight")
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
