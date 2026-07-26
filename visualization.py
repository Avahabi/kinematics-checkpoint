"""Matplotlib visualization helpers for the toy biped."""

from __future__ import annotations

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def draw_box(ax, link, color=(0.38, 0.68, 0.95), edgecolor="k") -> None:
    """Draw a link box using its local vertices transformed into world coordinates."""
    if link.vertex is None or link.face is None:
        return

    vertices = (link.R @ link.vertex.T).T + link.p
    polygons = [[vertices[index] for index in face] for face in link.face]
    collection = Poly3DCollection(polygons, facecolors=[color], edgecolors=edgecolor, linewidths=0.7, alpha=0.95)
    ax.add_collection3d(collection)


def draw_cylinder_joint(ax, pos: np.ndarray, radius: float = 0.025, color=(0.08, 0.22, 0.42)) -> None:
    """Draw a small spherical marker for a joint."""
    u = np.linspace(0.0, 2.0 * np.pi, 16)
    v = np.linspace(0.0, np.pi, 8)
    x = pos[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = pos[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = pos[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, linewidth=0, shade=True)


def draw_all_joints(ax, links: list, link_id: int = 1) -> None:
    """Recursively draw all joints and visible link geometry."""
    if link_id == 0:
        return

    link = links[link_id]
    draw_box(ax, link)

    if link.mother != 0:
        parent = links[link.mother]
        xs = [parent.p[0], link.p[0]]
        ys = [parent.p[1], link.p[1]]
        zs = [parent.p[2], link.p[2]]
        ax.plot(xs, ys, zs, color="k", linewidth=2.0)

    draw_cylinder_joint(ax, link.p)

    draw_all_joints(ax, links, link.child)
    draw_all_joints(ax, links, link.sister)


def set_biped_axes(ax, show_labels: bool = True) -> None:
    """Set axes limits and view for the toy biped figure."""
    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(-0.35, 0.35)
    ax.set_zlim(0.1, 1.3)
    ax.set_box_aspect((0.9, 0.7, 1.2))
    ax.view_init(elev=13, azim=46, roll=0)
    ax.grid(True)
    ax.set_xlabel("x" if show_labels else "")
    ax.set_ylabel("y" if show_labels else "")
    ax.set_zlabel("z" if show_labels else "")
