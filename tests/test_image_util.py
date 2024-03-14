from config.constants import DataPath
from src.processors.SobelFilter import SobelFilter
from src.processors.sc.ConnectedSC import ConnectedSC
from src.processors.sc.DisconnectedSC import DisconnectedSC
from src.processors.sc.ThreadedSC import ThreadedSC
from utils.Image import Image
from config.plotter import Plotter

name = "img_5.png"

PATH = f"{DataPath.INPUT_PATH.value}/{name}"

img_rgb = Image(PATH, gray=False)
img_gray = Image(PATH, gray=True)()

img_new = img_rgb.rgb()
img_org = img_rgb.rgb()

# energy = CannyProcessor(img_gray)().image()
energy = SobelFilter(img_gray)().image()

# Plotter.image(energy)

ratio = 0.75

img_new1, energy1 = ThreadedSC(img_new, energy, ratio)()
# img_new2, energy2 = ConnectedSC(img_new, energy, ratio)()
# img_new, energy = DisconnectedSC(img_new, energy, ratio)()

Image.save(img_new, name)

Plotter.images([img_org, img_new1, energy1], 1, 3)
