import numpy as np

from config.constants import DataPath
from config.plotter import Plotter
from src.processors.SaliencyMap import SaliencyMap
from src.processors.SobelFilter import SobelFilter
from src.processors.sc.ForwardSCV2 import ForwardSCV2
from src.processors.sc.MiddleSC import MiddleSC
from util.Image import Image

dir = f"{DataPath.INPUT_PATH.value}/depth"

left_path = f"{dir}/left.jpg"
right_path = f"{dir}/right.jpg"

org = Image(left_path)

left = org.rgb()
gray = org.gray()
right = Image(right_path, gray=True)()

right = np.array(right)

right = abs(255 - right)

saliency = SobelFilter(gray)().image()

energy = right + saliency

# print(np.max(right))

result = ForwardSCV2(left, energy, 0.75)()

Plotter.images([left, energy, saliency, right, result], 2, 3)
