import os
import time
import seam_carving

from config.constants import DataPath
from config.plotter import Plotter
from src.processors.SaliencyMap import SaliencyMap
from src.processors.SobelFilter import SobelFilter
from src.processors.sc.ConnectedSC import ConnectedSC
from src.processors.sc.ForwardSC import ForwardSC
from src.processors.sc.MiddleSC import MiddleSC
from utils.Image import Image

path = DataPath.INPUT_PATH.value

paths = os.listdir(path)

images = []

total_time = 0
total_width = 0

total_time_b = 0
total_width_b = 0

count = 0
for i in paths:
    filePath = f"{path}/{i}"

    if not filePath.endswith(".png"):
        continue

    count += 1
    img = Image(filePath, gray=False)

    img_rgb = img.rgb()
    img_gray = img.gray()

    sobel = SobelFilter(img_gray)().image()
    saliency = SaliencyMap(img, 0.6)().image()

    # energy = grab_cut_energy(img, img_gray)
    energy = saliency + sobel

    # forward = ForwardSC(img_rgb, energy, 0.75)()

    # Image.save(forward, f"forward/{i}")

    # backward = ConnectedSC(img_rgb, energy, 0.75)()

    # Image.save(backward, f"backward/{i}")

    start = time.perf_counter()
    MiddleSC(img_rgb, energy, 0.75)()
    end = time.perf_counter()

    elapsed_time = end - start
    total_time += elapsed_time
    total_width += img_rgb.shape[1]

    seams = int(0.25 * img_rgb.shape[1])

    start = time.perf_counter()
    seam_carving.resize(
        img_rgb, (seams, img_rgb.shape[0]),
        energy_mode='backward',  # Choose from {backward, forward}
        order='width-first',  # Choose from {width-first, height-first}
        keep_mask=None
    )
    end = time.perf_counter()

    elapsed_time = end - start
    total_time_b += elapsed_time
    total_width_b += img_rgb.shape[1]

    # print(f"{i} => {img_rgb.shape}, time: {elapsed_time:.4f}")

    # Plotter.images([img_rgb, energy, result], 2, 2)

    # break

# Plotter.images(images, 1, len(paths))

print("time", total_time / count)
print("width", total_width / count)

print("time", total_time_b / count)
print("width", total_width_b / count)
