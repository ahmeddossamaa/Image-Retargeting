from config.constants import DataPath
from src.SobelFilterProcessor import SobelFilterProcessor
from src.CannyProcessor import CannyProcessor
from utils.Image import Image
from utils.Plotter import Plotter

img = Image(f"{DataPath.INPUT_PATH.value}/img.png")()
img2 = img.copy()

sobel = SobelFilterProcessor(img)()
canny = CannyProcessor(img2)()
Plotter.image("Sobel", sobel.image())
Plotter.image("Canny", canny.image())
