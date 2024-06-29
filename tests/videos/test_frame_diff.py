import numpy as np

from config.constants import DataPath
from config.plotter import Plotter
from src.processors.Combiner import Combiner
from src.processors.sc.MiddleSCI import MiddleSCI
from utils.Image import Image

category = "ball"

PATH = f"../{DataPath.OUTPUT_PATH.value}/frames/{category}"

frame1 = Image(f"{PATH}/50.jpg")
frame2 = Image(f"{PATH}/52.jpg")

ratio = 0.75

# Plotter.images([frame1.rgb(), frame2.rgb()], 1, 2)

r1 = MiddleSCI(frame1, ratio, converter=Combiner)()

r2 = MiddleSCI(frame2, ratio, converter=Combiner)()

Plotter.images([r1, r2], 1, 2)

# Plotter.images([frame1, frame2, frame2 - frame1], 1, 3)
