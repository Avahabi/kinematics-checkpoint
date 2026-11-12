"""Step through numerical IK while the right foot follows a Bezier trajectory."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from inverse_kinematics import Pose, inverse_kinematics
from kinematics import rpy_to_rot
from robot_model import LLEG_J5, RLEG_J5, go_half_sitting, setup_biped_robot
from visualization import draw_all_joints, set_biped_axes


def quadratic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, s: float) -> np.ndarray:
    """Return one point on a quadratic Bezier curve."""
    return (1 - s) ** 2 * p0 + 2 * (1 - s) * s * p1 + s ** 2 * p2


def make_right_foot_target(s: float, initial_p: np.ndarray) -> Pose:
    """Build the desired right-foot pose at normalized path coordinate s."""
    to_rad = np.pi / 180.0

    intermediate_p = initial_p + np.array([0.2, 0.0, 0.30])
    final_p = initial_p + np.array([0.4, 0.0, 0.20])

    position = quadratic_bezier(initial_p, intermediate_p, final_p, s)

    initial_pitch = 0.0 * to_rad
    intermediate_pitch = -10.0 * to_rad
    final_pitch = -20.0 * to_rad

    pitch = (1 - s) ** 2 * initial_pitch + 2 * (1 - s) * s * intermediate_pitch + s ** 2 * final_pitch

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
    """Create a light copy of the link list for animation frames."""
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


def solve_bezier_motion_with_copies(num_points: int = 35) -> tuple[list[list], np.ndarray, list[float]]:
    """Solve IK and store independent robot states for each animation frame."""
    links = setup_biped_robot()
    go_half_sitting(links)

    initial_right_foot_p = links[RLEG_J5].p.copy()
    initial_left_foot_p = links[LLEG_J5].p.copy()
    left_foot = make_left_foot_target(initial_left_foot_p)

    solved_frames = []
    path_points = []
    errors = []

    for s in np.linspace(0.0, 1.0, num_points):
        right_foot = make_right_foot_target(float(s), initial_right_foot_p)

        left_error = inverse_kinematics(links, LLEG_J5, left_foot)
        right_error = inverse_kinematics(links, RLEG_J5, right_foot)

        solved_frames.append(copy_links(links))
        path_points.append(right_foot.p.copy())
        errors.append(float(max(np.max(np.abs(left_error)), np.max(np.abs(right_error)))))

    return solved_frames, np.array(path_points), errors


def show_step_viewer() -> None:
    """Open a Matplotlib viewer with buttons for stepping through the IK result."""
    frames, path_points, errors = solve_bezier_motion_with_copies()
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
        ax.set_title(f"Bezier IK frame {frame_id + 1}/{len(frames)} | error {errors[frame_id]:.2e}")
        ax.legend(loc="upper left")
        figure.canvas.draw_idle()

    def step_backward(_event) -> None:
        state["frame_id"] = max(0, state["frame_id"] - 1)
        draw_current_frame()

    def step_forward(_event) -> None:
        state["frame_id"] = min(len(frames) - 1, state["frame_id"] + 1)
        draw_current_frame()

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

# Verification answers:
# 1. Using the measured foot position as P0 makes the first target equal the actual pose, so there is no jump at the start.
# 2. No. P1 is a control point that pulls the curve; the curve only passes through P0 and P2.
# 3. The previous solution is close to the next one, so the solver converges faster and stays on the same continuous branch.
# 4. The links list is mutable and changes on every IK call; without copies, every frame would show the last state.
# 5. The left foot is the stance/support foot in contact with the ground.
