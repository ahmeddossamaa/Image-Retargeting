import numpy as np
import seam_carving

from config.constants import DataPath
from config.decorators import Decorators
from config.plotter import Plotter
from src.processors.SobelFilter import SobelFilter
from utils.Image import Image

name = "img_6.png"

PATH = f"{DataPath.INPUT_PATH.value}/{name}"

img_gray = Image(PATH, gray=False).rgb()

energy = SobelFilter(img_gray)().image()


@Decorators.Loggers.log_method_time
def fun():
    src = np.array(img_gray)
    src_h, src_w, _ = src.shape
    dst = seam_carving.resize(
        src, (src_w - 106, src_h),
        energy_mode='backward',  # Choose from {backward, forward}
        order='width-first',  # Choose from {width-first, height-first}
        keep_mask=None
    )

    return dst


dst = fun()

Plotter.image(dst)
