from config.constants import DataPath
from src.SobelFilterProcessor import SobelFilterProcessor
from utils.Image import Image
from utils.Plotter import Plotter

img = Image(f"{DataPath.INPUT_PATH.value}/img.png")()

sobel = SobelFilterProcessor(img)()

Plotter.image(sobel.image())
