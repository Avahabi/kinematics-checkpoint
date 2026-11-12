"""Pose numerical IK on RPO using Pinocchio, then visualize in Panda3D."""

from __future__ import annotations

import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Pose:
    p: np.ndarray
    R: np.ndarray


def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])

    return rz @ ry @ rx


def create_urdf_with_absolute_mesh_paths(urdf_path: Path) -> Path:
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("filename")
        if not filename or filename.startswith("package://"):
            continue

        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh_path = (urdf_path.parent / mesh_path).resolve()

        mesh.attrib["filename"] = mesh_path.as_posix()

    temp_file = tempfile.NamedTemporaryFile(mode="wb", suffix=".urdf", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    tree.write(temp_path, encoding="utf-8", xml_declaration=True)
    return temp_path


def find_frame_id(model, candidates: list[str]) -> int:
    frame_names = [frame.name for frame in model.frames]

    for candidate in candidates:
        if candidate in frame_names:
            return model.getFrameId(candidate)

    print("Could not find any candidate foot frame.")
    print()
    print("Available frames containing ankle, foot, leg, or toe:")
    for name in frame_names:
        low = name.lower()
        if any(word in low for word in ["ankle", "foot", "leg", "toe"]):
            print(" ", name)

    sys.exit(1)


def get_joint_q_index(model, joint_name: str) -> int:
    joint_id = model.getJointId(joint_name)

    if joint_id >= model.njoints:
        print(f"Required joint not found: {joint_name}")
        sys.exit(1)

    if model.nqs[joint_id] != 1:
        print(f"Joint is not 1-DoF: {joint_name}")
        sys.exit(1)

    return model.idx_qs[joint_id]


def get_joint_v_index(model, joint_name: str) -> int:
    joint_id = model.getJointId(joint_name)

    if joint_id >= model.njoints:
        print(f"Required joint not found: {joint_name}")
        sys.exit(1)

    if model.nvs[joint_id] != 1:
        print(f"Joint velocity is not 1-DoF: {joint_name}")
        sys.exit(1)

    return model.idx_vs[joint_id]


def solve_pose_ik_right_leg(
    pin,
    model,
    data,
    q,
    frame_id: int,
    target: Pose,
    right_leg_joint_names: list[str],
    max_iters: int = 400,
    damping: float = 1e-3,
    alpha: float = 0.6,
    joint_limit_rad: float = 1.5,
    orientation_weight: float = 0.05,
):
    q_indices = [get_joint_q_index(model, name) for name in right_leg_joint_names]
    v_indices = [get_joint_v_index(model, name) for name in right_leg_joint_names]

    def compute_error(q_test):
        pin.forwardKinematics(model, data, q_test)
        pin.updateFramePlacements(model, data)

        current_pose = data.oMf[frame_id]
        current_p = current_pose.translation
        current_R = current_pose.rotation

        orientation_error = pin.log3(target.R @ current_R.T)

        position_error = target.p - current_p

        error = np.concatenate([position_error, orientation_weight * orientation_error])

        return error, np.linalg.norm(error), current_p, current_R, position_error, orientation_error

    for iteration in range(max_iters):
        (
            pose_error,
            error_norm,
            current_p,
            current_R,
            position_error,
            orientation_error,
        ) = compute_error(q)

        print(f"iteration {iteration:03d} | pose error norm: {error_norm:.6f}")

        if error_norm < 1e-4:
            return q, True, error_norm

        full_jacobian = pin.computeFrameJacobian(
            model,
            data,
            q,
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        task_jacobian = full_jacobian[:, v_indices].copy()
        task_jacobian[3:, :] *= orientation_weight

        J = task_jacobian
        dq_leg = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), pose_error)

        accepted = False

        for scale in [alpha, alpha * 0.5, alpha * 0.25, alpha * 0.1, alpha * 0.05]:
            q_candidate = q.copy()

            for i, qi in enumerate(q_indices):
                q_candidate[qi] = np.clip(
                    q_candidate[qi] + scale * dq_leg[i], -joint_limit_rad, joint_limit_rad
                )

            candidate_norm = compute_error(q_candidate)[1]
            if candidate_norm < error_norm:
                q = q_candidate
                accepted = True
                break

        if not accepted:
            print("  update rejected: no smaller step reduced the error")
            return q, False, error_norm

    return q, False, error_norm


def main() -> None:
    try:
        import pinocchio as pin
        from pinocchio.visualize import Panda3dVisualizer
    except ModuleNotFoundError as error:
        print("A required package is not installed.")
        print(f"Missing module: {error.name}")
        sys.exit(1)

    robot_pkg = Path.home() / "kinematics" / "robot_models" / "rpo_description"
    urdf_path = robot_pkg / "urdf" / "rpo.urdf"

    if not urdf_path.exists():
        print("URDF file not found:")
        print(urdf_path)
        sys.exit(1)

    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()

    q = pin.neutral(model)

    q[get_joint_q_index(model, "right_knee_joint")] = 0.90
    q[get_joint_q_index(model, "right_ankle_pitch_joint")] = -0.5

    right_foot_frame = find_frame_id(
        model,
        [
            "right_ankle_roll_link",
            "right_ankle_link",
            "right_foot",
            "right_foot_link",
            "r_foot",
            "r_foot_link",
        ],
    )

    right_leg_joint_names = [
        "right_thigh_yaw_joint",
        "right_thigh_roll_joint",
        "right_thigh_pitch_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ]

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    initial_pose = data.oMf[right_foot_frame]
    initial_p = initial_pose.translation.copy()
    initial_R = initial_pose.rotation.copy()

    to_rad = np.pi / 180.0

    right_foot = Pose(
        p=initial_p + np.array([0.6, 0.0, 0.1]),
        R=initial_R @ rpy_to_rot(0.0, -30.0 * to_rad, 0.0),
    )

    print("RPO pose numerical IK test")
    print("URDF:", urdf_path)
    print("Target frame:", model.frames[right_foot_frame].name)
    print()
    print("Right-leg joints used by IK:")
    for name in right_leg_joint_names:
        print(" ", name)
    print()
    print("Initial target position:")
    print(np.round(initial_p, 4))
    print("Desired target position:")
    print(np.round(right_foot.p, 4))
    print()

    q_solution, success, final_error_norm = solve_pose_ik_right_leg(
        pin,
        model,
        data,
        q.copy(),
        right_foot_frame,
        right_foot,
        right_leg_joint_names,
        orientation_weight=0.05,
    )

    pin.forwardKinematics(model, data, q_solution)
    pin.updateFramePlacements(model, data)

    final_pose = data.oMf[right_foot_frame]
    final_p = final_pose.translation.copy()
    final_R = final_pose.rotation.copy()

    final_position_error = right_foot.p - final_p
    final_orientation_error = pin.log3(right_foot.R @ final_R.T)

    print()
    print("IK success:", success)
    print("Final pose error norm:", f"{final_error_norm:.6f}")
    print()
    print("Final target position:")
    print(np.round(final_p, 4))
    print("Final position error [m]:")
    print(np.round(final_position_error, 6))
    print("Final orientation error [rad]:")
    print(np.round(final_orientation_error, 6))
    print()
    print("Right-leg joint solution [rad]:")
    for name in right_leg_joint_names:
        q_index = get_joint_q_index(model, name)
        print(f"{name:30s} {q_solution[q_index]: .4f}")

    patched_urdf_path = create_urdf_with_absolute_mesh_paths(urdf_path)

    visual_model = pin.buildGeomFromUrdf(
        model,
        str(patched_urdf_path),
        pin.GeometryType.VISUAL,
    )
    collision_model = pin.GeometryModel()

    viz = Panda3dVisualizer(model, collision_model, visual_model)
    viz.initViewer()
    viz.loadViewerModel("rpo_pose_ik")
    viz.display(q_solution)

    print()
    print("Panda3D is displaying the RPO pose-IK result.")
    print("Close the Panda3D window, then press Enter here to finish.")
    input()


if __name__ == "__main__":
    main()

# Verification answers:
# 1. A target relative to the measured pose is reachable and consistent with the robot's real geometry and frame placement.
# 2. IK converges to the solution nearest the initial guess, so the positive knee guess selects that knee-bend branch.
# 3. Weighting both keeps the error and the Jacobian consistent; otherwise the update solves a mismatched problem.
# 4. Near singularities J J^T is ill-conditioned; damping prevents huge, unstable joint steps.
# 5. A full step can overshoot in this nonlinear problem; smaller scales help find a step that actually reduces the error.
