# -------
# runs the specified vla model below
# -------

import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from PIL import Image
from xarm import XArmAPI 

# Me implemented
from camera import Camera

PROMPT = "In: What action should the robot take to put the blue block in the cardboard box?\nOut:"
IP = "192.168.1.198"

# I need to quantize the model to run it
bnb = BitsAndBytesConfig(
    load_in_8bit=True,
)

# copied model -- OpenVLA base no fine tuning
from transformers import AutoModelForVision2Seq
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    quantization_config=bnb,
    device_map="auto",
)

# this is for 
processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b", 
    trust_remote_code=True,
)

# ROBOT
robot = XArmAPI(IP)
robot.motion_enable(True)
robot.set_mode(0)
robot.set_state(0)
robot.set_gripper_enable(True)

# camera
camera = Camera()

# LOOP for RUNNING
for step in range(100):
    image: Image.Image = camera.get_frame()
    inputs = processor(PROMPT, image).to("cuda:0", dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=True)
 
    print(action)

    dx, dy, dz, dr, dp, dyaw, grip = action  # m, m, m, rad, rad, rad, [0,1]
    robot.set_position(x=dx*1000, y=dy*1000, z=dz*1000, roll=dr*57.2958, pitch=dp*57.2958, yaw=dyaw*57.2958, relative=True, speed=50, wait=True)
    robot.set_gripper_position(850 if grip > 0.5 else 0, wait=False)
