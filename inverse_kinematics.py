"""Numerical inverse kinematics helpers for the toy biped."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kinematics import forward_kinematics, rot_to_omega


@dataclass
class Pose:
    """Desired position and attitude of a link."""

    p: np.ndarray
    R: np.ndarray


def find_route(links: list, to_link: int) -> list[int]:
    """Return the joint IDs from the body link to the target link."""
    parent = links[to_link].mother
    if parent == 1:
        return [to_link]
    return [*find_route(links, parent), to_link]


def calc_jacobian(links: list, route: list[int]) -> np.ndarray:
    """Calculate the 6D geometric Jacobian in the world frame."""
    target = links[route[-1]].p
    jacobian = np.zeros((6, len(route)))

    for col, link_id in enumerate(route):
        link = links[link_id]
        axis_world = link.R @ link.a
        jacobian[:3, col] = np.cross(axis_world, target - link.p)
        jacobian[3:, col] = axis_world

    return jacobian


def calc_pose_error(target: Pose, current) -> np.ndarray:
    """Return stacked position and angular error for one link."""
    position_error = target.p - current.p
    rotation_error = current.R.T @ target.R
    angular_error = current.R @ rot_to_omega(rotation_error)
    return np.concatenate([position_error, angular_error])


def move_joints(links: list, route: list[int], delta_q: np.ndarray) -> None:
    """Apply a joint-angle update to the selected route."""
    for link_id, delta in zip(route, delta_q):
        links[link_id].q += float(delta)


def inverse_kinematics(
    links: list,
    to_link: int,
    target: Pose,
    gain: float = 0.9,
    max_iterations: int = 10,
    tolerance: float = 1e-6,
) -> float:
    """Solve IK by repeatedly applying a Jacobian-based joint update."""
    route = find_route(links, to_link)
    forward_kinematics(links, 1)
    error = calc_pose_error(target, links[to_link])

    for _ in range(max_iterations):
        if np.linalg.norm(error) < tolerance:
            break
        jacobian = calc_jacobian(links, route)
        try:
            delta_q = gain * np.linalg.solve(jacobian, error)
        except np.linalg.LinAlgError:
            delta_q = gain * np.linalg.lstsq(jacobian, error, rcond=None)[0]
        move_joints(links, route, delta_q)
        forward_kinematics(links, 1)
        error = calc_pose_error(target, links[to_link])

    return float(np.linalg.norm(error))
