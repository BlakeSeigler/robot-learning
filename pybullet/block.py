import time
from pathlib import Path

import pybullet as p


def main():
    
    urdf_path = Path(__file__).parent / "deps" / "uf-gym" / "urdf" / "xarm" / "xarm6_with_gripper.urdf"

    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)

    ground_col = p.createCollisionShape(p.GEOM_PLANE)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=ground_col)

    p.loadURDF(str(urdf_path), useFixedBase=True)

    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025], rgbaColor=[1, 0.2, 0.2, 1])
    p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=[0.4, 0, 0.025])

    time.sleep(10)
    p.disconnect()


if __name__ == "__main__":
    main()
