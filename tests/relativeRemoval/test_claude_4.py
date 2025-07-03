import numpy as np
from seam_carving import resize

from config.constants import DataPath
from config.decorators import Decorators
from config.plotter import Plotter
from src.processors.Composer import Composer
from src.processors.Fuser import Fuser
from src.processors.sc.MiddleSCI import MiddleSCI
from src.processors.sc.v2.ProgressiveSeamCarver import ProgressiveSeamCarver
from src.processors.sc.v2.RegionBasedSeamCarver import RegionBasedSeamCarver
from src.processors.sc.v2.SeamCarver import SeamCarver
from util.Image import Image

@Decorators.Loggers.log_method_time
def main():
    maps = []
    images = []
    paths = ["Baby_santa"]

    for i in paths:
        img = Image(f"../{DataPath.INPUT_PATH.value}/cair/{i}.jpg")

        images.append(img)

        fuser = Fuser(img)()

        print("shapes:", img.rgb().shape, fuser.image().shape)

        maps.append(img.rgb())
        maps.append(fuser.image())
        maps.append(fuser.get_depth())
        # maps.append(fuser.get_lines())
        # maps.append(fuser.get_orientation())
        maps.append(fuser.get_saliency())
        maps.append(fuser.get_gradient())

        break

    Plotter.images(maps, 2, 3)

    idx = 0

    img = images[idx]
    energy = maps[idx * 5 + 1]

    image = img.rgb()

    src = np.array(image)

    src_h, src_w, _ = image.shape

    size_mask = (int(0.75 * src_w), src_h)

    carver = SeamCarver(image, energy)

    result = carver.seamCarve(0.75)

    backward_lib = resize(
        src, size_mask,
        energy_mode='backward',
        order='width-first',
        keep_mask=energy
    )

    forward_lib = resize(
        src, size_mask,
        energy_mode='forward',
        order='width-first',
        keep_mask=energy
    )

    Plotter.images([image, energy, result, backward_lib, forward_lib], 2, 3)


if __name__ == "__main__":
    main()