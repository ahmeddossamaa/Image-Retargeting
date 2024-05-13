import numpy as np

import cv2

from config.constants import DataPath
from config.plotter import Plotter
from utils.Image import Image

i = 3

name = f"sgbm"

PATH = f"{DataPath.INPUT_PATH.value}/{name}"

img = Image(f"{PATH}/right.jpg", gray=False)
img_rgb = img.rgb()
left = Image(f"{PATH}/left.jpg", gray=True)()
right = Image(f"{PATH}/right.jpg", gray=True)()

window_size = 5
min_disp = 0
nDispFactor = 10
num_disp = 16 * nDispFactor - min_disp

# stereo = cv2.StereoSGBM().create(
#     minDisparity=min_disp,
#     numDisparities=num_disp,
#     blockSize=window_size,
#     P1=8*3*window_size**2,
#     P2=32*3*window_size**2,
#     disp12MaxDiff=1,
#     uniquenessRatio=15,
#     speckleWindowSize=0,
#     speckleRange=2,
#     preFilterCap=63,
#     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
# )

# stereo = cv2.StereoBM().create(numDisparities=num_disp, blockSize=window_size)
#
# disparity = stereo.compute(left, right).astype(float) / 16.0

images = []

images.append(left)
images.append(right)

for i in range(7):
    stereo = cv2.StereoBM().create(numDisparities=16 * (i + 1), blockSize=window_size)

    disparity = stereo.compute(left, right).astype(float) / 16.0

    images.append(disparity)

Plotter.images(images, 3, 3)
