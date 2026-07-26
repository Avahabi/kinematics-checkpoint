"""Demonstrate numerical inverse kinematics on the toy biped."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from inverse_kinematics import Pose, inverse_kinematics
from kinematics import rpy_to_rot
from robot_model import LLEG_J5, RLEG_J5, go_half_sitting, setup_biped_robot
from visualization import draw_all_joints, set_biped_axes


def solve_two_foot_targets() -> tuple[list, float, float]:
    """Solve two foot targets with Jacobian-based numerical IK."""
    to_rad = np.pi / 180.0
    links = setup_biped_robot()
    go_half_sitting(links)

    right_foot = Pose(
        p=np.array([0.2, -0.1, 0.4]),
        R=rpy_to_rot(0.0, -10.0 * to_rad, 0.0),
    )
    right_error = inverse_kinematics(links, RLEG_J5, right_foot)

    left_foot = Pose(
        p=np.array([0.0, 0.11, 0.0]),
        R=rpy_to_rot(0.0, 0.0 * to_rad, 0.0),
    )
    left_error = inverse_kinematics(links, LLEG_J5, left_foot)

    return links, right_error, left_error


def draw_result() -> None:
    links, right_error, left_error = solve_two_foot_targets()

    figure = plt.figure(figsize=(6, 6))
    ax = figure.add_subplot(111, projection="3d")
    draw_all_joints(ax, links, 1)
    set_biped_axes(ax)
    ax.set_title("Numerical IK result")
    plt.show()


def main() -> None:
    draw_result()


if __name__ == "__main__":
    main()
