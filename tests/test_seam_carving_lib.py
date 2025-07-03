import numpy as np
import seam_carving

from config.constants import DataPath
from config.decorators import Decorators
from config.plotter import Plotter
from src.processors.Combiner import Combiner
from src.processors.Fuser import Fuser
from src.processors.SobelFilter import SobelFilter
from src.processors.sc.ImprovedSC import ImprovedSC
from src.processors.sc.MiddleSCI import MiddleSCI
from src.processors.sc.v2.DynamicSC import DynamicSC
from src.processors.sc.v2.MiddleForwardSC import MiddleForwardSC
from util.Image import Image

name = "cair/Baby_santa.jpg"

PATH = f"{DataPath.INPUT_PATH.value}/{name}"

img = Image(PATH)

energy = Combiner(img)().image()


@Decorators.Loggers.log_method_time
def fun():
    src = np.array(img.rgb())
    src_h, src_w, _ = img.rgb().shape
    dst = seam_carving.resize(
        src, (src_w - int(src_w / 4), src_h),
        energy_mode='forward',  # Choose from {backward, forward}
        order='width-first',  # Choose from {width-first, height-first}
        keep_mask=None
    )

    return dst


dst = fun()
# middle = MiddleSCI(img, 0.75, converter=Combiner)()
# improved = MiddleForwardSC(img, 0.75, converter=Fuser)()
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

Plotter.images([img.rgb(), dyn, dst], 1, 3)

# strategies = ['adaptive_change', 'energy_variance', 'seam_quality', 'hybrid']
# results = [img.rgb()]
# for strategy in strategies:
#     results.append(DynamicSC(
#         img,
#         ratio=0.75,
#         strategy=strategy,
#         converter=Fuser,
#         energy_threshold=0.15,
#         min_interval=3,
#         max_interval=12,
#         change_threshold=0.12
#     )())
#
# Plotter.images(results, 2, 3)