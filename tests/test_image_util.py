from config.constants import DataPath
from src.SeamCarving import SeamCarving
from src.SobelFilter import SobelFilter
from src.CannyProcessor import CannyProcessor
from utils.Image import Image
from config.plotter import Plotter

PATH = f"{DataPath.INPUT_PATH.value}/img_1.png"

img = Image(PATH, gray=True)()
img_rgb = Image(PATH, gray=False)()

Plotter.image(img_rgb)

# img = img.to_gray_scale()
temp = img_rgb.copy()
# for i in range(50):
img = SobelFilter(img)().image()

temp, img = SeamCarving(is_connected=False)(temp, img, 1, 1)

Plotter.images([img_rgb, temp], 1, 2)
