from shutil import rmtree

import numpy as np
from seam_carving import resize

from src.processors.Fuser import Fuser
from src.processors.sc.v2.SeamCarver import SeamCarver
from util.Image import Image
from os import listdir, path, mkdir, rmdir

from config.constants import DataPath

dataPath = "./../../../Datasets/cair"

folders = listdir(dataPath)

count = 0
totalCount = len(folders)

print(totalCount)

for i in folders:
    imgPath = path.join(dataPath, i)

    if not path.exists(imgPath):
        continue

    i = i[:len(i) - 4]

    print(i)

    outputPath = f"./../{DataPath.OUTPUT_PATH.value}/cair_processed/{i}"

    if path.exists(outputPath):
        rmtree(outputPath)

    mkdir(outputPath)

    img = Image(imgPath)

    rgb = img.rgb()

    src = np.array(rgb)

    src_h, src_w, _ = rgb.shape

    size_mask = (int(0.75 * src_w), src_h)

    fuser = Fuser(img)()

    backward_lib = resize(
        src, size_mask,
        energy_mode='backward',
        order='width-first',
        keep_mask=None
    )

    forward_lib = resize(
        src, size_mask,
        energy_mode='forward',
        order='width-first',
        keep_mask=None
    )

    result = SeamCarver(img.rgb(), fuser.image()).seamCarve(0.75)

    Image.save(img.rgb(), f"{outputPath}/0-origin.jpg")
    Image.save(result, f"{outputPath}/1-result.jpg")
    Image.save(backward_lib, f"{outputPath}/2-backward_lib.jpg")
    Image.save(forward_lib, f"{outputPath}/3-forward_lib.jpg")
    Image.save((fuser.image() * 255).astype(np.uint8), f"{outputPath}/4-processed.jpg")
    Image.save((fuser.get_depth() * 255).astype(np.uint8), f"{outputPath}/5-depth.jpg")
    Image.save((fuser.get_saliency() * 255).astype(np.uint8), f"{outputPath}/6-saliency.jpg")
    Image.save((fuser.get_gradient() * 255).astype(np.uint8), f"{outputPath}/7-gradient.jpg")