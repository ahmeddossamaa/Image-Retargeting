from config.constants import DataPath
from config.decorators import Decorators
from src.SeamCarving import SeamCarving
from src.SobelFilter import SobelFilter
from src.CannyProcessor import CannyProcessor
from utils.Image import Image
from config.plotter import Plotter

PATH = f"{DataPath.INPUT_PATH.value}/img_3.png"

img = Image(PATH, gray=True)()
img_rgb = Image(PATH, gray=False)

img_new = img_rgb()
img_org = img_rgb()

# energy = CannyProcessor(img)().image()
energy = SobelFilter(img)().image()

# Plotter.image(energy)

img_new, energy = SeamCarving(is_connected=True)(img_new, energy, 50, 1)

Plotter.images([img_org, img_new], 1, 2)
