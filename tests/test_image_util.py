from config.constants import DataPath
from src.processors.SeamCarving import SeamCarving
from src.processors.SobelFilter import SobelFilter
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

img_new, energy = SeamCarving(is_connected=True)(img_new, energy, 100, 1)

Image.save(img_new, "img_4.png")

Plotter.images([img_org, img_new], 1, 2)
