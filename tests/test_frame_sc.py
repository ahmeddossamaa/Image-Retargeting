import numpy as np

from config.constants import DataPath
from config.plotter import Plotter
from src.processors.Combiner import Combiner
from src.processors.SaliencyMap import SaliencyMap
from src.processors.SobelFilter import SobelFilter
from src.processors.sc.ImprovedSC import ImprovedSC
from src.processors.sc.MiddleSC import MiddleSC
from utils.Image import Image

if __name__ == '__main__':
    name = "frames/ball/50.jpg"
    # name = "img_6.png"

    img = Image(f"{DataPath.OUTPUT_PATH.value}/{name}")

    rgb = img.rgb()
    # gray = img.gray()
    #
    # saliency = SaliencyMap(img)().image()
    # sobel = SobelFilter(gray)().image()

    # energy = np.maximum(sobel / 255, saliency)

    energy = Combiner(img)().image()

    result = ImprovedSC(img, 0.75, converter=Combiner)()

    # result = MiddleSC(rgb, energy, 0.75)()

    Plotter.images([rgb, energy, result], 1, 3)
