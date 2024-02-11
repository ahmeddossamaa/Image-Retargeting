from config.constants import DataPath
from config.plotter import Plotter
from src.CannyProcessor import CannyProcessor
from src.SobelFilter import SobelFilter
from utils.Image import Image

img = Image(f"{DataPath.INPUT_PATH.value}/img_2.png")()
img2 = img.copy()

sobel = SobelFilter(img)().image()
canny = CannyProcessor(img2)().image()

# Plotter.image(sobel.image(), "Sobel")
# Plotter.image(canny.image(), "Canny")

Plotter.images([sobel, canny], 1, 2)
