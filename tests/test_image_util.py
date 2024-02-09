from config.constants import DataPath
from src.SeamCarving import SeamCarving
from src.SobelFilter import SobelFilter
from utils.Image import Image
from config.plotter import Plotter

PATH = f"{DataPath.INPUT_PATH.value}/img.png"

img = Image(PATH, gray=True)()
img_rgb = Image(PATH, gray=False)()

# Plotter.image(img_rgb)

# img = img.to_gray_scale()

img = SobelFilter(img)().image()

img = SeamCarving(is_connected=False)(img_rgb, img, 1, 1)

# print(img.shape)

Plotter.images([img_rgb, img], 1, 2)
