from config.constants import DataPath
from config.plotter import Plotter
from src.processors.sc.ForwardSCV2 import ForwardSCV2
from util.Image import Image

i = 5

name = f"img_{i}.png"

PATH = f"{DataPath.INPUT_PATH.value}/{name}"

img = Image(PATH, gray=False)
img_rgb = img.rgb()
img_gray = Image(PATH, gray=True)()

result = ForwardSCV2(img_rgb, img_gray, 0.75)()

Plotter.images([img_rgb, result], 1, 2)
