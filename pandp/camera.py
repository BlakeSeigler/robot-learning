import pyrealsense2 as rs
import numpy as np
from PIL import Image


class Camera:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self.pipeline.start(config)

    def get_frame(self) -> Image.Image:
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        return Image.fromarray(np.asarray(color.get_data()))

    def stop(self):
        self.pipeline.stop()
