from config.constants import DataPath
from config.plotter import Plotter
from src.processors.CannyProcessor import CannyProcessor
from src.processors.SaliencyMap import SaliencyMap
from src.processors.SobelFilter import SobelFilter
from utils.Image import Image


images = []

for i in range(6, 10):
    name = f"img_{i + 1}.png"

    PATH = f"{DataPath.INPUT_PATH.value}/{name}"

    img = Image(PATH, gray=False)
    img_rgb = img.rgb()
    img_gray = Image(PATH, gray=True)()

    images.append(img_rgb)
    saliency = SaliencyMap(img)().image()
    images.append(saliency)
    # print(saliency)
    images.append(SobelFilter(saliency)().image())
    images.append(CannyProcessor(img_gray)().image())


Plotter.images(images, 4, 4)
