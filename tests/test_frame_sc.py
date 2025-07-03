from config.constants import DataPath
from config.plotter import Plotter
from src.processors.Combiner import Combiner
from src.processors.SaliencyMap import SaliencyMap
from src.processors.SobelFilter import SobelFilter
# from src.processors.Fuser import Fuser
from src.processors.sc.MiddleSCI import MiddleSCI
from src.processors.sc.v2.MiddleForwardSC import MiddleForwardSC
from util.Image import Image

if __name__ == '__main__':
    # name = "frames/ball/50.jpg"
    # name = "img_4.png"

    path = f"{DataPath.INPUT_PATH.value}"

    img = Image(f"{path}/bedroom.png")
    rgb = img.rgb()

    # energyMap = Fuser(img)().image()
    #
    # Plotter.images([rgb, energyMap], 2, 1)

    middle = MiddleSCI(img, 0.75, converter=Combiner)()

    # result = MiddleForwardSC(img, 0.75, converter=SobelFilter)()

    Plotter.images([rgb, middle], 1, 2)
