import numpy as np
from rembg import remove

from config.constants import DataPath
from config.plotter import Plotter
from utils.Image import Image

n = 5

name = f"img_{n}.png"

PATH = f"{DataPath.INPUT_PATH.value}/{name}"

image = Image(PATH)
gray = image.gray()
rgb = image.rgb()

output = remove(rgb)

# output = np.array(output)
# print(np.min(output))
print(output)

output = np.where((output == [255, 255, 255]).all(axis=2), [0, 0, 0], [255, 255, 255])

print(output)
print(np.min(output))

Plotter.images([rgb, output], 1, 2)
