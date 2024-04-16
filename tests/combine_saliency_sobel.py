import math

import numpy as np
from cv2 import threshold, CV_8U, THRESH_BINARY, connectedComponentsWithStats
from matplotlib import pyplot as plt
import seaborn as sns

from config.constants import DataPath
from src.processors.sc.ConnectedSC import ConnectedSC
from src.processors.sc.ForwardSC import ForwardSC
from utils.Image import Image
from config.plotter import Plotter
from src.processors.SaliencyMap import SaliencyMap
from src.processors.SobelFilter import SobelFilter


n = 5

name = f"img_{n}.png"

PATH = f"{DataPath.INPUT_PATH.value}/{name}"

img = Image(PATH, gray=False)
img_rgb = img.rgb()
img_gray = Image(PATH, gray=True)()

ts = 0.35

old_saliency = SaliencyMap(img)().image()
sobel = SobelFilter(img_gray)().image()

_, saliency = threshold(old_saliency, ts, 1, THRESH_BINARY)

height, width, z = img_rgb.shape

energy = np.maximum(sobel / 255, saliency)

print(sobel)
print(old_saliency)
print(saliency)

isOff = True

# Plotter.images([sobel, old_saliency, saliency], 1, 3, off=False)

backward = ConnectedSC(img_rgb, old_saliency, 0.75)()
forward = ForwardSC(img_rgb, old_saliency, 0.75)()

Image.save(backward, f"backward-{ts}-{name}")
Image.save(forward, f"forward-{ts}-{name}")

Plotter.images([
    img_rgb, old_saliency,
    backward, forward
], 2, 2, off=False)
