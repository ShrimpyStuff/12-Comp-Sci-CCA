"""Geodesic dome variant without the top apex node.

This module reuses the existing geodesic dome generator, then removes the
single apex node and stores the ring of nodes that were connected to the apex
in ``apex_id`` so downstream code can treat the top ring as the new apex
group. The bottom ring is preserved as ``base_ids``.
"""
from collections import namedtuple

import numpy as np

try:
    import geodesic
except ImportError:
    import src.geodesic as geodesic


OpenDome = namedtuple("OpenDome", "nodes members base_ids apex_id")


def _remap_indices(keep_mask):
    remap = -np.ones(len(keep_mask), dtype=int)
    remap[keep_mask] = np.arange(int(keep_mask.sum()))
    return remap


def generate_open_dome(R, h, V, radial_offsets=None):
    """Generate a geodesic dome without the apex node.

    The nodes that were adjacent to the original apex are returned in apex_id.
    """
    dome = geodesic.generate_dome(R=R, h=h, V=V, radial_offsets=radial_offsets)

    keep_nodes = np.ones(len(dome.nodes), dtype=bool)
    keep_nodes[dome.apex_id] = False
    remap = _remap_indices(keep_nodes)

    nodes = dome.nodes[keep_nodes].copy()

    members = []
    apex_ids = []
    seen_apex = set()
    for n1, n2 in dome.members:
        if dome.apex_id in (n1, n2):
            other = n2 if n1 == dome.apex_id else n1
            mapped_other = int(remap[other])
            if mapped_other not in seen_apex:
                seen_apex.add(mapped_other)
                apex_ids.append(mapped_other)
            continue
        members.append((int(remap[n1]), int(remap[n2])))

    base_ids = [int(remap[i]) for i in dome.base_ids]
    return OpenDome(nodes=nodes, members=members, base_ids=base_ids, apex_id=apex_ids)


def visualize_open_dome(dome, title="Open geodesic dome", savepath=None, ax=None):
    """Render the open dome with apex-ring nodes and base ring highlighted."""
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

    if dome.apex_id:
        tn = nodes[dome.apex_id]
        ax.scatter(tn[:, 0], tn[:, 1], tn[:, 2], c="orange", s=30,
                   label=f"apex ring ({len(dome.apex_id)} nodes)")

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

    dome = generate_open_dome(R=5.0, h=5.0, V=2)
    visualize_open_dome(dome, title="Open geodesic dome V=2", savepath="open_dome_V2.png")
