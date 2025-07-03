from shutil import rmtree, copy

import numpy as np

from seam_carving import resize
from config.plotter import Plotter
from src.processors.Combiner import Combiner
from src.processors.Fuser import Fuser
from src.processors.SaliencyMap import SaliencyMap
from src.processors.sc.BackwardSC import BackwardSC
from src.processors.sc.ImprovedSC import ImprovedSC
from src.processors.sc.MiddleSCI import MiddleSCI
from src.processors.sc.v2.DynamicSC import DynamicSC
from src.processors.sc.v2.MiddleForwardSC import MiddleForwardSC
from util.Image import Image
from os import listdir, path, mkdir, rmdir

from config.constants import DataPath

dataPath = "./../../Datasets/cair"

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

    outputPath = f"./{DataPath.OUTPUT_PATH.value}/cair/{i}"

    if path.exists(outputPath):
        rmtree(outputPath)

    mkdir(outputPath)
    mkdir(f"{outputPath}/maps")

    img = Image(imgPath)

    rgb = img.rgb()
    gray = img.gray()

    # energy = SaliencyMap(img)().image()

    featureMap = Fuser(img)()

    ratio = 0.75

    src = np.array(rgb)
    src_h, src_w, _ = rgb.shape

    size_mask = (int(0.75 * src_w), src_h)

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

    backward = BackwardSC(rgb, featureMap.getSaliency(), ratio)()
    forward = ImprovedSC(img, ratio, feature_map=featureMap.getSaliency())()
    middle = MiddleSCI(img, ratio, feature_map=featureMap.image())()
    dyn = DynamicSC(
        img,
        ratio=0.75,
        converter=Fuser,
        # strategy='seam_quality',
        energy_threshold=0.15,
        min_interval=3,
        max_interval=12,
        change_threshold=0.12
    )()
    # batched = MiddleForwardSC(img, ratio, feature_map=featureMap.image())()

    # copy(f"{dataPath}/{i}/1-saliencymap.png", f"{outputPath}/maps/saliency.png")
    # copy(f"{dataPath}/{i}/2-Sobel.png", f"{outputPath}/maps/sobel.png")
    # copy(f"{dataPath}/{i}/3-Depthmap.png", f"{outputPath}/maps/depth.png")
    # copy(f"{dataPath}/{i}/5-output-middleSC.png", f"{outputPath}/forward-middle.png")
    # if path.exists(f"{dataPath}/{i}/6-output- multi_scale_fusion_result.png"):
    #     copy(f"{dataPath}/{i}/6-output- multi_scale_fusion_result.png", f"{outputPath}/maps/multiscale-fusion.png")

    Image.save(rgb, f"{outputPath}/1-original.png")
    Image.save(backward_lib, f"{outputPath}/2-backward-library.png")
    Image.save(forward_lib, f"{outputPath}/3-forward-library.png")
    Image.save(backward, f"{outputPath}/4-backward-saliency.png")
    Image.save(forward, f"{outputPath}/5-forward-saliency.png")
    Image.save(middle, f"{outputPath}/6-forward-middle.png")
    Image.save(dyn, f"{outputPath}/7-forward-middle-adaptive.png")

    # Image.save(SaliencyMap, f"{outputPath}/maps/SaliencyMap.png")
    Image.save(featureMap.image(), f"{outputPath}/maps/fusion.png")
    Image.save(featureMap.getDepth(), f"{outputPath}/maps/depth.png")
    Image.save(featureMap.getSaliency(), f"{outputPath}/maps/saliency.png")
    Image.save(featureMap.getGradient(), f"{outputPath}/maps/gradient.png")

    # Plotter.images([
    #     rgb,
    #     backward_lib,
    #     forward_lib,
    #     backward,
    #     forward,
    #     dyn,
    #     # batched
    # ], 2, 3)

    count += 1

    print(f"========> {float(count/totalCount) * 100}%")
