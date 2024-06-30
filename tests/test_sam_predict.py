import cv2
import numpy as np
from segment_anything import SamPredictor

from config.constants import DataPath
from config.plotter import Plotter
from util.Image import Image
from util.Sam import sam

mask_predictor = SamPredictor(sam)

name = "img_3.png"

PATH = f"{DataPath.INPUT_PATH.value}/{name}"

image_rgb = Image(PATH).rgb()

mask_predictor.set_image(image_rgb)

box = np.array([70, 247, 626, 926])
masks, scores, logits = mask_predictor.predict(
    box=box,
    multimask_output=True
)

# print(masks)
# print(scores)

Plotter.images(masks, 1, 3)
