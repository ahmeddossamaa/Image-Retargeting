import numpy as np

from config.constants import DataPath
from config.plotter import Plotter
from utils.Image import Image

category = "ball"

PATH = f"{DataPath.OUTPUT_PATH.value}/frames/{category}"

frame1 = Image(f"{PATH}/50.jpg").gray()
frame2 = Image(f"{PATH}/60.jpg").gray()

Plotter.images([frame1, frame2, frame2 - frame1], 1, 3)

# Plotter.images([frame1, frame2], 1, 2)
