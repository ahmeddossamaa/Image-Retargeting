from config.constants import DataPath
from src.SeamCarving import SeamCarving
from src.SobelFilter import SobelFilter
from utils.Image import Image
from config.plotter import Plotter

img = Image(f"{DataPath.INPUT_PATH.value}/img.png", gray=True)()

img = SobelFilter(img)().image()

img = SeamCarving(connected=False)(img, img, 1, 1)

Plotter.image(img)
