from config.constants import DataPath
from config.plotter import Plotter
from src.processors.OptimalFusionProcessor import OptimalFusionProcessor
from src.processors.sc.MiddleSCI import MiddleSCI
from util.Image import Image

name = "cair/Beach.jpg"

PATH = f"{DataPath.INPUT_PATH.value}/{name}"

def test_fusion_strategies(image_path):
    """Test different fusion strategies and return results"""
    from util.Image import Image

    img = Image(image_path)

    strategies = {
        'proximity_priority': OptimalFusionProcessor(img, 'proximity_priority'),
        'adaptive': OptimalFusionProcessor(img, 'adaptive'),
        'detail_preserving': OptimalFusionProcessor(img, 'detail_preserving'),
        'custom_balanced': OptimalFusionProcessor(img, 'custom', [0.4, 0.4, 0.2])
    }

    results = []
    for name, processor in strategies.items():
        results.append(processor().image())

    return results

# Plotter.images(test_fusion_strategies(PATH), 2, 2)

images = test_fusion_strategies(PATH)

# for i in images:
#     MiddleSCI(img, 0.75, feature_map=)()

img = Image(PATH)

results = []
for image in images:
    results.append(MiddleSCI(img, 0.75, feature_map=image)())

Plotter.images(results, 2, 2)