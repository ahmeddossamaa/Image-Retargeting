from config.constants import DataPath
from config.plotter import Plotter
from src.processors.SaliencyMap import SaliencyMap
from src.processors.sc.ConnectedSC import ConnectedSC
from utils.Image import Image

PATH = DataPath.INPUT_PATH.value

name = "img_3.png"

image = Image(f"{PATH}/{name}")

image_rgb = image.rgb()

# Plotter.images([
#     image.rgb(),
#     image.gray(),
#     image.lab(),
# ], 1, 3, off=True)

saliency_map = SaliencyMap(image)().image()

# Plotter.image(saliency_map)

result = ConnectedSC(image_rgb, saliency_map, 0.50, color=False)()

Plotter.images([image_rgb, saliency_map, result], 1, 3)
