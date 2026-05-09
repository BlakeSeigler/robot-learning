import math
import sys
import time

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    p = None
    pybullet_data = None


class OpenVLMSim:
    """Tiny stand-in for a vision-language model policy."""

    def decide(self, observation, instruction):
        block_x, block_y, _ = observation["block_pos"]
        target_x, target_y, _ = observation["target_pos"]
        distance = math.hypot(target_x - block_x, target_y - block_y)

        if "move" in instruction.lower() and distance > 0.05:
            return "move block to target"
        return "done"


def main():
    if p is None:
        run_text_sim()
    else:
        run_pybullet_sim()


def run_text_sim():
    policy = OpenVLMSim()
    instruction = "move the red block to the green target"
    block_pos = [0, 0, 0.03]
    target_pos = [0.35, 0.2, 0.03]

    for _ in range(40):
        observation = {"block_pos": block_pos, "target_pos": target_pos}
        action = policy.decide(observation, instruction)
        print(f"{action}: block={block_pos[0]:.2f},{block_pos[1]:.2f}")

        if action == "done":
            return

        block_pos[0] += (target_pos[0] - block_pos[0]) * 0.18
        block_pos[1] += (target_pos[1] - block_pos[1]) * 0.18
        time.sleep(0.05)


def run_pybullet_sim():
    use_gui = "--gui" in sys.argv
    p.connect(p.GUI if use_gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    block = p.loadURDF("cube_small.urdf", [0, 0, 0.03])
    target_pos = [0.35, 0.2, 0.03]
    target = p.loadURDF("cube_small.urdf", target_pos, globalScaling=1.2)
    p.changeVisualShape(block, -1, rgbaColor=[0.9, 0.1, 0.1, 1])
    p.changeVisualShape(target, -1, rgbaColor=[0.1, 0.8, 0.2, 0.35])

    policy = OpenVLMSim()
    instruction = "move the red block to the green target"

    for _ in range(120):
        block_pos, _ = p.getBasePositionAndOrientation(block)
        observation = {"block_pos": block_pos, "target_pos": target_pos}
        action = policy.decide(observation, instruction)
        print(action)

        if action == "done":
            break

        x = block_pos[0] + (target_pos[0] - block_pos[0]) * 0.06
        y = block_pos[1] + (target_pos[1] - block_pos[1]) * 0.06
        p.resetBasePositionAndOrientation(block, [x, y, 0.03], [0, 0, 0, 1])
        p.stepSimulation()
        time.sleep(1 / 60)

    time.sleep(1)
    p.disconnect()


if __name__ == "__main__":
    main()
