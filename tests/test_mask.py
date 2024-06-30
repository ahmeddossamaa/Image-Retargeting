from cv2 import threshold, THRESH_BINARY

from config.constants import DataPath
from config.plotter import Plotter
from src.processors.sc.ConnectedSC import ConnectedSC
from util.Image import Image

name = "img_3.png"

PATH = f"{DataPath.INPUT_PATH.value}/{name}"

img = Image(PATH, gray=False)

img_rgb = img.rgb()
img_gray = img.gray()

_, mask = threshold(img_gray, 128, 255, THRESH_BINARY)

Plotter.image(mask, off=True)

result = ConnectedSC(img_rgb, mask, 0.75)()

# print(result)

Plotter.images([img_rgb, mask, result], 1, 3)
