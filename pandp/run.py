# -------
# runs the specified vla model below
# -------

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from xarm import XArmAPI 

# Me implemented
from camera import Camera

PROMPT = "Pick up the ..."
IP = "192.168.1.222"

# copied model -- OpenVLA base no fine tuning
from transformers import AutoModelForVision2Seq
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", 
    trust_remote_code=True, 
    dtype="auto"
).to("cuda:0")

# this is for 
processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b", 
    trust_remote_code=True,
)

# ROBOT
robot = XArmAPI(IP)

# camera
camera = Camera()

# LOOP for RUNNING
for step in range(100):
    image: Image.Image = camera.get_from_camera()
    inputs = processor(PROMPT, image).to("cuda:0", dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=True)
    robot.move(action)
