import numpy as np
import cv2

from config.constants import DataPath
from config.decorators import Decorators
from config.plotter import Plotter
from src.processors.CannyProcessor import CannyProcessor
from src.processors.SaliencyMap import SaliencyMap
from src.processors.SobelFilter import SobelFilter
from src.processors.sc.ConnectedSC import ConnectedSC
from src.processors.sc.ForwardSC import ForwardSC
from util.Image import Image

n = 2

name = f"img_{n}.png"

PATH = f"{DataPath.INPUT_PATH.value}/{name}"

image = Image(PATH, gray=False)
img_rgb = image.rgb()
img_gray = image.gray()

height, width, z = img_rgb.shape

_, result = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# saliency = SaliencyMap(image, 0.3)().image()
# sobel = SobelFilter(img_gray)().image()

# print(result, saliency)

# energy = result - saliency * 255

# Plotter.images([saliency, result, energy], 1, 3)


@Decorators.Loggers.log_method_time
def grab_cut_energy(img, gray):
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    mask = SobelFilter(gray)().image()

    rect = (0, 0, width - 1, height - 1)
    cv2.grabCut(img_rgb, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Plotter.image(mask2)

    mask3 = SaliencyMap(img)().image()

    Plotter.images([mask, mask2, mask3], 1, 3)

    img = mask2 + mask3 * 255.0

    return img


energy = grab_cut_energy(image, img_gray)

# Plotter.image(energy)

backward = ConnectedSC(img_rgb, energy, 0.75)()
forward = ForwardSC(img_rgb, energy, 0.75)()

Plotter.images([img_rgb, energy, backward, forward], 2, 2)
