import os
import threading
import time
import datetime
import numpy as np
import tensorflow as tf
from pynput import keyboard
from xarm import XArmAPI
from camera import Camera

IP = "192.168.1.198"
HZ = 5
XYZ_MM = 5
ROT_DEG = 5.0
SAVE_DIR = "episodes"

last_action = np.zeros(7, dtype=np.float32)
stop_flag = threading.Event()


def encode_step(step, is_first, is_last):
    img = tf.io.encode_jpeg(tf.constant(step["image"])).numpy()
    return tf.train.Example(features=tf.train.Features(feature={
        "observation/image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
        "observation/state": tf.train.Feature(float_list=tf.train.FloatList(value=step["state"])),
        "action":            tf.train.Feature(float_list=tf.train.FloatList(value=step["action"])),
        "reward":            tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
        "is_first":          tf.train.Feature(int64_list=tf.train.Int64List(value=[int(is_first)])),
        "is_last":           tf.train.Feature(int64_list=tf.train.Int64List(value=[int(is_last)])),
        "is_terminal":       tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
    }))


def save_episode(buf):
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = f"{SAVE_DIR}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.tfrecord"
    with tf.io.TFRecordWriter(path) as w:
        for i, step in enumerate(buf):
            w.write(encode_step(step, i == 0, i == len(buf) - 1).SerializeToString())
    print(f"Saved {len(buf)} steps → {path}")


def sampler(robot, camera, buf):
    interval = 1.0 / HZ
    while not stop_flag.is_set():
        t = time.time()
        frame = np.array(camera.get_frame())
        _, pos = robot.get_position()
        buf.append({
            "image": frame,
            "state": np.array(pos[:6], dtype=np.float32),
            "action": last_action.copy(),
        })
        time.sleep(max(0, interval - (time.time() - t)))


def controller(robot):
    global last_action

    def on_press(key):
        global last_action
        dx = dy = dz = dr = dp = dyaw = 0.0
        grip = last_action[6]

        if key == keyboard.Key.up:      dx = XYZ_MM
        elif key == keyboard.Key.down:  dx = -XYZ_MM
        elif key == keyboard.Key.right: dy = XYZ_MM
        elif key == keyboard.Key.left:  dy = -XYZ_MM
        elif key == keyboard.Key.space: dz = -XYZ_MM

        try:
            c = key.char
            if   c == 'v': dz = XYZ_MM
            elif c == 'a': dr = ROT_DEG
            elif c == 's': dr = -ROT_DEG
            elif c == 'e': dp = ROT_DEG
            elif c == 'r': dp = -ROT_DEG
            elif c == 'd': dyaw = ROT_DEG
            elif c == 'f': dyaw = -ROT_DEG
            elif c == 'g':
                grip = 0.0
                robot.set_gripper_position(0, wait=False)
            elif c == 'h':
                grip = 1.0
                robot.set_gripper_position(850, wait=False)
            elif c == 'q':
                stop_flag.set()
                return False
        except AttributeError:
            pass

        if any([dx, dy, dz, dr, dp, dyaw]):
            robot.set_position(x=dx, y=dy, z=dz, roll=dr, pitch=dp, yaw=dyaw,
                               relative=True, speed=50, wait=False)

        last_action = np.array([
            dx / 1000, dy / 1000, dz / 1000,
            np.radians(dr), np.radians(dp), np.radians(dyaw),
            grip,
        ], dtype=np.float32)

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


def main():
    global last_action, stop_flag

    robot = XArmAPI(IP)
    robot.motion_enable(True)
    robot.set_mode(0)
    robot.set_state(0)
    robot.set_gripper_enable(True)
    camera = Camera()

    try:
        while True:
            input("\nPress Enter to start episode (q to stop and save)...")
            buf = []
            last_action = np.zeros(7, dtype=np.float32)
            last_action[6] = 1.0  # start with gripper open
            stop_flag = threading.Event()

            t = threading.Thread(target=sampler, args=(robot, camera, buf), daemon=True)
            t.start()
            controller(robot)
            t.join(timeout=1.0)

            if buf:
                save_episode(buf)

    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
