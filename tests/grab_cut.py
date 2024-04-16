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
from utils.Image import Image

n = 7

name = f"img_{n}.png"

PATH = f"{DataPath.INPUT_PATH.value}/{name}"

image = Image(PATH, gray=False)
img_rgb = image.rgb()
img_gray = image.gray()

height, width, z = img_rgb.shape

_, result = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

saliency = SaliencyMap(image, 0.3)().image()
sobel = SobelFilter(img_gray)().image()

# print(result, saliency)

energy = result - saliency * 255

Plotter.images([saliency, result, energy], 1, 3)


@Decorators.Loggers.log_method_time
def fun():
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # mask = SaliencyMap(image, 0.8)().image()
    mask = CannyProcessor(image)().image()

    temp = mask

    rect = (0, 0, width // 2, height)
    cv2.grabCut(img_rgb, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Plotter.image(mask)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Plotter.images([temp, mask, mask2], 1, 3)

    mask3 = SaliencyMap(image)().image()

    print(mask2)
    print(mask3)

    # img = img_rgb * mask2[:, :, np.newaxis]
    img = mask2 + mask3

    print("max", np.max(mask2))

    return img


# energy = fun()
#
# Plotter.image(energy)
#
# backward = ConnectedSC(img_rgb, energy, 0.75)()
# forward = ForwardSC(img_rgb, energy, 0.75)()
#
# Plotter.images([img_rgb, backward, forward], 1, 3)
