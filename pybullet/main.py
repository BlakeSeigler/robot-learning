import time
from pathlib import Path

import pybullet as p


def main():
    
    urdf_path = Path(__file__).parent / "deps" / "uf-gym" / "urdf" / "xarm" / "xarm6_with_gripper.urdf"

    p.connect(p.GUI)
    p.loadURDF(str(urdf_path), useFixedBase=True)
    time.sleep(100)
    p.disconnect()


if __name__ == "__main__":
    main()
