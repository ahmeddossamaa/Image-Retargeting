import numpy as np
from PIL import Image as PILImage
from transformers import pipeline
import torch
import cv2

from config.helper import Helper
from util.Processor import Processor
from util.Image import Image


class DepthAnythingV2(Processor):
    def __init__(self, image: Image):
        super().__init__(image)
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)

    def main(self, *args, **kwargs):
        image_array = self._image()

        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)

        pil_image = PILImage.fromarray(image_array)
        depth = self.pipe(pil_image)["depth"]
        depth_array = np.array(depth)

        depth_normalized = Helper.Math.normalize(depth_array, 0, 1)
        # _, depth_binary = cv2.threshold(depth_normalized, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        self._image = depth_normalized