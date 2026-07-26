"""Kinematics helpers for the toy humanoid biped."""

from __future__ import annotations

import numpy as np


def hat(vector: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix for a 3D vector."""
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )


def rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return the rotation matrix for a rotation around axis by angle radians."""
    axis = np.asarray(axis, dtype=float).reshape(3)
    norm_axis = np.linalg.norm(axis)
    if norm_axis < np.finfo(float).eps:
        return np.eye(3)

    unit_axis = axis / norm_axis
    theta = norm_axis * angle
    axis_hat = hat(unit_axis)
    return (
        np.eye(3)
        + axis_hat * np.sin(theta)
        + axis_hat @ axis_hat * (1.0 - np.cos(theta))
    )


def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return the rotation matrix for roll, pitch, yaw angles."""
    c_phi = np.cos(roll)
    s_phi = np.sin(roll)
    c_the = np.cos(pitch)
    s_the = np.sin(pitch)
    c_psi = np.cos(yaw)
    s_psi = np.sin(yaw)

    return np.array(
        [
            [c_psi * c_the, -s_psi * c_phi + c_psi * s_the * s_phi, s_psi * s_phi + c_psi * s_the * c_phi],
            [s_psi * c_the, c_psi * c_phi + s_psi * s_the * s_phi, -c_psi * s_phi + s_psi * s_the * c_phi],
            [-s_the, c_the * s_phi, c_the * c_phi],
        ]
    )


def rot_to_omega(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix into the equivalent rotation vector."""
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    element = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    norm_element = np.linalg.norm(element)
    if norm_element > np.finfo(float).eps:
        angle = np.arctan2(norm_element, np.trace(rotation) - 1.0)
        return angle / norm_element * element
    if rotation[0, 0] > 0.0 and rotation[1, 1] > 0.0 and rotation[2, 2] > 0.0:
        return np.zeros(3)
    return 0.5 * np.pi * np.array(
        [
            rotation[0, 0] + 1.0,
            rotation[1, 1] + 1.0,
            rotation[2, 2] + 1.0,
        ]
    )


def find_mothers(links: list) -> None:
    """Fill each link's mother field from child/sister links."""
    for link in links:
        if link is not None:
            link.mother = 0

    def visit(parent_id: int) -> None:
        child_id = links[parent_id].child
        while child_id != 0:
            links[child_id].mother = parent_id
            visit(child_id)
            child_id = links[child_id].sister

    visit(1)


def forward_kinematics(links: list, link_id: int = 1) -> None:
    """Recursively update world position and attitude for all links."""
    if link_id == 0:
        return

    link = links[link_id]
    if link_id != 1:
        mother = links[link.mother]
        link.p = mother.R @ link.b + mother.p
        link.R = mother.R @ rodrigues(link.a, link.q)

    forward_kinematics(links, link.sister)
    forward_kinematics(links, link.child)
