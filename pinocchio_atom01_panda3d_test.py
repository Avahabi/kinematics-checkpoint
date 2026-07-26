"""Load RoboParty Atom01/RPO from URDF, analyze it with Pinocchio, and view it in Panda3D."""

from __future__ import annotations

import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def create_urdf_with_absolute_mesh_paths(urdf_path: Path) -> Path:
    """Create a temporary URDF where relative mesh paths become absolute paths."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue

        if filename.startswith("package://"):
            continue

        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh_path = (urdf_path.parent / mesh_path).resolve()

        mesh.attrib["filename"] = mesh_path.as_posix()

    temp_file = tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".urdf",
        delete=False,
    )
    temp_path = Path(temp_file.name)
    temp_file.close()

    tree.write(temp_path, encoding="utf-8", xml_declaration=True)
    return temp_path


def main() -> None:
    try:
        import pinocchio as pin
        from pinocchio.visualize import Panda3dVisualizer
    except ModuleNotFoundError as error:
        print("A required package is not installed.")
        print(f"Missing module: {error.name}")
        sys.exit(1)

    model_root = Path.home() / "kinematics" / "robot_models"
    robot_pkg = model_root / "rpo_description"
    urdf_path = robot_pkg / "urdf" / "rpo.urdf"

    if not urdf_path.exists():
        print("URDF file not found:")
        print(urdf_path)
        sys.exit(1)

    patched_urdf_path = create_urdf_with_absolute_mesh_paths(urdf_path)

    root_joint = pin.JointModelFreeFlyer()

    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        str(patched_urdf_path),
        root_joint=root_joint,
    )

    data = model.createData()

    q = pin.neutral(model)
    v = np.zeros(model.nv)

    # Lift the floating base so the robot appears above the grid.
    q[2] = 0.75

    pin.forwardKinematics(model, data, q, v)
    pin.updateFramePlacements(model, data)

    com = pin.centerOfMass(model, data, q, v)
    com_jacobian = pin.jacobianCenterOfMass(model, data, q)

    print("RoboParty Atom01/RPO loaded from URDF")
    print("Original URDF:", urdf_path)
    print("Temporary URDF:", patched_urdf_path)
    print(f"configuration dimension nq: {model.nq}")
    print(f"velocity dimension nv:      {model.nv}")
    print(f"number of joints:           {model.njoints}")
    print(f"number of visual objects:   {len(visual_model.geometryObjects)}")
    print()
    print("Joint names:")
    for name in model.names:
        print(" ", name)
    print()
    print("Center of mass [m]:")
    print(np.round(com, 4))
    print()
    print("Center of mass Jacobian shape:")
    print(com_jacobian.shape)
    print()

    viz = Panda3dVisualizer(model, collision_model, visual_model)
    viz.initViewer()
    viz.loadViewerModel("atom01_rpo")
    viz.display(q)

    print("Panda3D is displaying the RoboParty Atom01/RPO URDF model.")
    print("Close the Panda3D window, then press Enter here to finish.")
    input()


if __name__ == "__main__":
    main()