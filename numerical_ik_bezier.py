"""Step through numerical IK while the right foot follows a Bezier trajectory."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from inverse_kinematics import Pose, inverse_kinematics
from kinematics import rpy_to_rot
from robot_model import (
    LLEG_J5,
    RLEG_J5,
    set_initial_bent_leg_posture,
    setup_biped_robot,
)
from visualization import draw_all_joints, set_biped_axes


def quadratic_bezier(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    s: float,
) -> np.ndarray:
    """Return one point on a quadratic Bezier curve."""
    # TODO 1: Implement B(s) using p0, p1, p2, and s.
    raise NotImplementedError


def make_right_foot_target(s: float, initial_p: np.ndarray) -> Pose:
    """Build the desired right-foot pose at normalized path coordinate s."""
    to_rad = np.pi / 180.0

    # TODO 2: Define the control and final positions relative to initial_p.
    intermediate_p = None
    final_p = None

    # TODO 3: Evaluate the Bezier position at s.
    position = None

    initial_pitch = 0.0 * to_rad
    intermediate_pitch = -10.0 * to_rad
    final_pitch = -20.0 * to_rad

    # TODO 4: Interpolate pitch with the same quadratic Bezier rule.
    pitch = None

    return Pose(
        p=position,
        R=rpy_to_rot(0.0, pitch, 0.0),
    )


def make_left_foot_target(initial_p: np.ndarray) -> Pose:
    """Build the fixed desired pose for the left support foot."""
    return Pose(
        p=initial_p.copy(),
        R=rpy_to_rot(0.0, 0.0, 0.0),
    )


def copy_links(links: list) -> list:
    """Create an independent copy of the robot state for one viewer frame."""
    copied = []
    for link in links:
        if link is None:
            copied.append(None)
            continue

        new_link = type(link)(
            name=link.name,
            mass=link.mass,
            sister=link.sister,
            child=link.child,
            b=link.b.copy(),
            a=link.a.copy(),
            q=link.q,
            mother=link.mother,
            p=link.p.copy(),
            R=link.R.copy(),
            vertex=None if link.vertex is None else link.vertex.copy(),
            face=None if link.face is None else link.face.copy(),
        )
        copied.append(new_link)
    return copied


def solve_bezier_motion(num_points: int = 35) -> tuple[list[list], np.ndarray, list[float]]:
    """Solve IK and store one independent robot state per path sample."""
    links = setup_biped_robot()
    set_initial_bent_leg_posture(links)

    initial_right_foot_p = links[RLEG_J5].p.copy()
    initial_left_foot_p = links[LLEG_J5].p.copy()
    left_foot = make_left_foot_target(initial_left_foot_p)

    solved_frames = []
    path_points = []
    errors = []

    for s in np.linspace(0.0, 1.0, num_points):
        right_foot = make_right_foot_target(float(s), initial_right_foot_p)

        # TODO 5: Solve the fixed left-foot target and moving right-foot target.
        left_error = None
        right_error = None

        # TODO 6: Store an independent robot state, target point, and largest error.

    return solved_frames, np.array(path_points), errors


def show_step_viewer() -> None:
    """Open a viewer with buttons for stepping through the IK result."""
    frames, path_points, errors = solve_bezier_motion()
    state = {"frame_id": 0}

    figure = plt.figure(figsize=(7, 6))
    ax = figure.add_subplot(111, projection="3d")
    figure.subplots_adjust(bottom=0.18)

    def draw_current_frame() -> None:
        frame_id = state["frame_id"]
        ax.cla()
        draw_all_joints(ax, frames[frame_id], 1)
        set_biped_axes(ax)
        ax.plot(
            path_points[:, 0],
            path_points[:, 1],
            path_points[:, 2],
            color="crimson",
            linewidth=2.0,
            label="right-foot Bezier path",
        )
        ax.scatter(
            path_points[frame_id, 0],
            path_points[frame_id, 1],
            path_points[frame_id, 2],
            color="crimson",
            s=35,
        )
        ax.set_title(
            f"Bezier IK frame {frame_id + 1}/{len(frames)} | "
            f"error {errors[frame_id]:.2e}"
        )
        ax.legend(loc="upper left")
        figure.canvas.draw_idle()

    def step_backward(_event) -> None:
        # TODO 7: Decrease frame_id without going below zero, then redraw.
        pass

    def step_forward(_event) -> None:
        # TODO 8: Increase frame_id without passing the last frame, then redraw.
        pass

    back_ax = figure.add_axes([0.30, 0.04, 0.16, 0.07])
    forward_ax = figure.add_axes([0.54, 0.04, 0.16, 0.07])
    back_button = Button(back_ax, "Backward")
    forward_button = Button(forward_ax, "Forward")
    back_button.on_clicked(step_backward)
    forward_button.on_clicked(step_forward)

    figure._bezier_buttons = (back_button, forward_button)
    draw_current_frame()
    plt.show()


def main() -> None:
    show_step_viewer()


if __name__ == "__main__":
    main()
