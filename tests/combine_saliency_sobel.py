import math

import numpy as np
from cv2 import threshold, CV_8U, THRESH_BINARY, connectedComponentsWithStats
from matplotlib import pyplot as plt
import seaborn as sns

from config.constants import DataPath
from src.processors.sc.ConnectedSC import ConnectedSC
from src.processors.sc.ForwardSCV2 import ForwardSCV2
from src.processors.sc.MiddleSC import MiddleSC
from src.processors.sc.ForwardSC import ForwardSC
from util.Image import Image
from config.plotter import Plotter
from src.processors.SaliencyMap import SaliencyMap
from src.processors.SobelFilter import SobelFilter


n = 2

name = f"boat/org.png"

PATH = f"{DataPath.INPUT_PATH.value}/{name}"

img = Image(PATH, gray=False)
img_rgb = img.rgb()
img_gray = Image(PATH, gray=True)()

ts = 0.4

old_saliency = SaliencyMap(img, ts)().image()
sobel = SobelFilter(img_gray)().image()

# _, saliency = threshold(old_saliency, ts, 1, THRESH_BINARY)

height, width, z = img_rgb.shape

energy = np.maximum(sobel / 255, old_saliency)
# energy = img_gray

# Plotter.image(energy)

# print(sobel)
# print(old_saliency)
# print(saliency)

isOff = True

# Plotter.images([sobel, old_saliency, saliency], 1, 3, off=False)

# backward = ConnectedSC(img_rgb, old_saliency, 0.75)()
# forward = ForwardSC(img_rgb, old_saliency, 0.75)()

# Image.save(backward, f"backward-{ts}-{name}")
# Image.save(forward, f"forward-{ts}-{name}")

# Plotter.images([
#     img_rgb, old_saliency,
#     backward, forward
# ], 2, 2, off=False)

middle = MiddleSC(img.rgb(), energy, ratio=0.75, color=True)()
forward = ForwardSCV2(img.rgb(), energy, ratio=0.75, color=True)()

Plotter.images([img_rgb, energy, middle, forward], 2, 2)
