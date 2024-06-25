import numpy as np
from cv2 import filter2D

from config.constants import DataPath
from config.plotter import Plotter
from src.processors.CannyProcessor import CannyProcessor
from src.processors.SaliencyMap import SaliencyMap
from src.processors.SobelFilter import SobelFilter
from src.processors.sc.MiddleSC import MiddleSC
from utils.Image import Image

n = 6

folder = "butterfly/"

name = f"{folder}org.png"
map_name = f"{folder}map.png"

PATH = f"{DataPath.INPUT_PATH.value}/{name}"
MAP_PATH = f"{DataPath.INPUT_PATH.value}/{map_name}"

img = Image(PATH, gray=False)
img_rgb = img.rgb()
img_gray = Image(PATH, gray=True)()

# saliency = Image(MAP_PATH, gray=True)()
saliency = SaliencyMap(img, 0.5)().image()

sobel = SobelFilter(img_gray)().image()
# sobel = SobelFilter(sobel)().image()
# sobel = SobelFilter(sobel)().image()

# canny = CannyProcessor(img_gray)().image()

# print(np.max(sobel))
# print(np.max(saliency))

energy = np.maximum(sobel, saliency * 255.0)
# energy = saliency

Plotter.images([img_rgb, sobel, energy], 1, 3)

result = MiddleSC(img_rgb, energy, 0.85, color=False)()

Image.save(result, "butterfly.png")

Plotter.images([img_rgb, result], 1, 2)

# # Define a simple 3x3 kernel for convolution (you can customize this)
# kernel = np.array([[0, -1, 0],
#                    [-1, 5, -1],
#                    [0, -1, 0]])
#
# # Apply convolution using cv2.filter2D
# for i in range(20):
#     img_gray = SobelFilter(img_gray)().image()
#
# Plotter.images([img_rgb, img.gray() - img_gray], 1, 3)
