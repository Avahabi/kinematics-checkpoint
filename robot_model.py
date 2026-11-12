"""Simple child-sister link-tree model for a toy biped."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kinematics import find_mothers, forward_kinematics


BODY = 1
RLEG_J0 = 2
RLEG_J1 = 3
RLEG_J2 = 4
RLEG_J3 = 5
RLEG_J4 = 6
RLEG_J5 = 7
LLEG_J0 = 8
LLEG_J1 = 9
LLEG_J2 = 10
LLEG_J3 = 11
LLEG_J4 = 12
LLEG_J5 = 13


@dataclass
class Link:
    name: str
    mass: float
    sister: int
    child: int
    b: np.ndarray
    a: np.ndarray
    q: float = 0.0
    mother: int = 0
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    vertex: np.ndarray | None = None
    face: np.ndarray | None = None


def make_box(dim: tuple[float, float, float], origin: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Create box vertices and faces for simple 3D link geometry."""
    x, y, z = dim
    ox, oy, oz = origin
    vertices = np.array(
        [
            [0, 0, 0],
            [0, y, 0],
            [x, y, 0],
            [x, 0, 0],
            [0, 0, z],
            [0, y, z],
            [x, y, z],
            [x, 0, z],
        ],
        dtype=float,
    )
    vertices -= np.array([ox, oy, oz], dtype=float)
    faces = np.array(
        [
            [0, 1, 2, 3],
            [1, 5, 6, 2],
            [3, 2, 6, 7],
            [0, 4, 7, 3],
            [0, 1, 5, 4],
            [4, 5, 6, 7],
        ],
        dtype=int,
    )
    return vertices, faces


def setup_biped_robot() -> list[Link | None]:
    """Create the toy biped link tree used by the random FK demo."""
    ux = np.array([1.0, 0.0, 0.0])
    uy = np.array([0.0, 1.0, 0.0])
    uz = np.array([0.0, 0.0, 1.0])

    links: list[Link | None] = [None] * 14
    links[BODY] = Link("BODY", 10, 0, 2, np.array([0.0, 0.0, 0.7]), uz)

    links[RLEG_J0] = Link("RLEG_J0", 5, 8, 3, np.array([0.0, -0.1, 0.0]), uz)
    links[RLEG_J1] = Link("RLEG_J1", 1, 0, 4, np.array([0.0, 0.0, 0.0]), ux)
    links[RLEG_J2] = Link("RLEG_J2", 5, 0, 5, np.array([0.0, 0.0, 0.0]), uy)
    links[RLEG_J3] = Link("RLEG_J3", 1, 0, 6, np.array([0.0, 0.0, -0.3]), uy)
    links[RLEG_J4] = Link("RLEG_J4", 6, 0, 7, np.array([0.0, 0.0, -0.3]), uy)
    links[RLEG_J5] = Link("RLEG_J5", 2, 0, 0, np.array([0.0, 0.0, 0.0]), ux)

    links[LLEG_J0] = Link("LLEG_J0", 5, 0, 9, np.array([0.0, 0.1, 0.0]), uz)
    links[LLEG_J1] = Link("LLEG_J1", 1, 0, 10, np.array([0.0, 0.0, 0.0]), ux)
    links[LLEG_J2] = Link("LLEG_J2", 5, 0, 11, np.array([0.0, 0.0, 0.0]), uy)
    links[LLEG_J3] = Link("LLEG_J3", 1, 0, 12, np.array([0.0, 0.0, -0.3]), uy)
    links[LLEG_J4] = Link("LLEG_J4", 6, 0, 13, np.array([0.0, 0.0, -0.3]), uy)
    links[LLEG_J5] = Link("LLEG_J5", 2, 0, 0, np.array([0.0, 0.0, 0.0]), ux)

    links[BODY].vertex, links[BODY].face = make_box((0.1, 0.3, 0.5), (0.05, 0.15, -0.05))
    links[RLEG_J5].vertex, links[RLEG_J5].face = make_box((0.2, 0.1, 0.02), (0.05, 0.05, 0.05))
    links[LLEG_J5].vertex, links[LLEG_J5].face = make_box((0.2, 0.1, 0.02), (0.05, 0.05, 0.05))

    find_mothers(links)
    links[BODY].p = np.array([0.0, 0.0, 0.65])
    links[BODY].R = np.eye(3)
    forward_kinematics(links, BODY)
    return links


def set_random_leg_angles(links: list[Link | None], rng: np.random.Generator) -> None:
    """Set random leg joint angles with limits similar to Fig. 2.24."""
    q_right = (2.0 / 3.0) * np.pi * (rng.random(6) - 0.5)
    q_right[3] = np.pi * rng.random()

    q_left = (2.0 / 3.0) * np.pi * (rng.random(6) - 0.5)
    q_left[3] = np.pi * rng.random()

    for offset in range(6):
        links[RLEG_J0 + offset].q = q_right[offset]
        links[LLEG_J0 + offset].q = q_left[offset]

    links[BODY].p = np.array([0.0, 0.0, 0.7])
    links[BODY].R = np.eye(3)
    forward_kinematics(links, BODY)


def go_half_sitting(links: list[Link | None]) -> None:
    """Move the toy biped to a non-singular half-sitting pose."""
    to_rad = np.pi / 180.0
    links[RLEG_J2].q = -5.0 * to_rad
    links[RLEG_J3].q = 10.0 * to_rad
    links[RLEG_J4].q = -5.0 * to_rad

    links[LLEG_J2].q = -5.0 * to_rad
    links[LLEG_J3].q = 10.0 * to_rad
    links[LLEG_J4].q = -5.0 * to_rad

    links[BODY].p = np.array([0.0, 0.0, 0.7])
    links[BODY].R = np.eye(3)
    forward_kinematics(links, BODY)
