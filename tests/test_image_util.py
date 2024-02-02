from config.constants import DataPath
from src.SeamCarving import SeamCarving
from src.SobelFilterProcessor import SobelFilterProcessor
from utils.Image import Image
from utils.Plotter import Plotter

img = Image(f"{DataPath.INPUT_PATH.value}/img.png")()

sobel = SobelFilterProcessor(img)().image()

result = SeamCarving()(img, sobel, 1, 1)

Plotter.image(sobel)
