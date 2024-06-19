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
    # name = "img_4.png"

    path = f"{DataPath.INPUT_PATH.value}/man1"

    img = Image(f"{path}/rgb.jpg")
    depth = Image(f"{path}/gt.jpg")()

    rgb = img.rgb()

    # print(depth.shape)

    # Plotter.images([rgb, depth], 1, 2)

    # gray = img.gray()
    #
    # saliency = SaliencyMap(img)().image()
    # sobel = SobelFilter(gray)().image()

    # energy = np.maximum(sobel / 255, saliency)

    energy = Combiner(img, depth=depth)().image()

    result = ImprovedSC(img, 0.750, converter=Combiner, feature_map=energy)()

    # result = MiddleSC(rgb, energy, 0.75)()

    Plotter.images([rgb, result], 1, 2)
