import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from config.constants import DataPath
from config.plotter import Plotter
from util.Image import Image
from util.Sam import sam

mask_generator = SamAutomaticMaskGenerator(sam)

# image_bgr = cv2.imread(IMAGE_PATH)
# image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

name = "img_6.png"

PATH = f"{DataPath.INPUT_PATH.value}/{name}"

image_rgb = Image(PATH).rgb()

result = mask_generator.generate(image_rgb)

print(result)

# Plotter.image(np.array(result))
