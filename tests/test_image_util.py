from config.constants import DataPath
from src.processors.SaliencyMap import SaliencyMap
from src.processors.SobelFilter import SobelFilter
from src.processors.sc.ConnectedSC import ConnectedSC
from src.processors.sc.ConnectedSCV2 import ConnectedSCV2
from src.processors.sc.DisconnectedSC import DisconnectedSC
from src.processors.sc.DisconnectedSCV2 import DisconnectedSCV2
from src.processors.sc.ThreadedSC import ThreadedSC
from utils.Image import Image
from config.plotter import Plotter

name = "img_3.png"

PATH = f"{DataPath.INPUT_PATH.value}/{name}"


img_rgb = Image(PATH, gray=False)
img_gray = Image(PATH, gray=True)()

img_new = img_rgb.rgb()
img_new2 = img_rgb.rgb()
img_org = img_rgb.rgb()

# energy = CannyProcessor(img_gray)().image()
# energy = SobelFilter(img_gray)().image()
energy = SaliencyMap(img_rgb)().image()

# Plotter.image(energy)

ratio = 0.75

# img_new = ThreadedSC(img_new, energy, ratio)()
img_new = ConnectedSC(img_new, energy, ratio, color=True)()
img_new2 = ConnectedSCV2(img_new2, energy, ratio, color=True)()
# img_new = DisconnectedSC(img_new, energy, ratio)()
# img_new = DisconnectedSCV2(img_new, energy, ratio)()

# Image.save(img_new, name)

Plotter.images([img_new, img_new2], 1, 2)
