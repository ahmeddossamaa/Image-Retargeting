from config.constants import DataPath
from src.SeamCarving import SeamCarving
from src.SobelFilterProcessor import SobelFilterProcessor
from utils.Image import Image
from utils.Plotter import Plotter
img = Image(f"{DataPath.INPUT_PATH.value}/img.png")()
from config.plotter import Plotter
import cv2

name = "img_20.png"


sobel = SobelFilterProcessor(img)().image()

result = SeamCarving()(img, sobel, 1, 1)

Plotter.image(sobel)
=======
img_new = img_rgb.rgb()
img_org = img_rgb.rgb()

# energy = CannyProcessor(img_gray)().image()
energy = SobelFilter(img_gray)().image()

# Plotter.image(energy)

ratio = 0.75

# img_new = ThreadedSC(img_new, energy, ratio)()
# img_new = ConnectedSC(img_new, energy, ratio, color=False)()
img_new = ConnectedSCV2(img_new, energy, ratio, color=False)()
# img_new = DisconnectedSC(img_new, energy, ratio)()
# img_new = DisconnectedSCV2(img_new, energy, ratio)()

Image.save(img_new, name)

Plotter.images([img_gray, img_new], 1, 2)
image_path_out = f"{DataPath.OUTPUT_PATH.value}/{name}"
cv2.imwrite(image_path_out, img_new)

