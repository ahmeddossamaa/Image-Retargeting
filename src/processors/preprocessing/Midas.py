import numpy as np
import torch
import torchvision
from torchvision import transforms

from config.constants import DataPath
from config.decorators import Decorators
from config.helper import Helper
from config.plotter import Plotter
from utils.Image import Image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(torch.cuda.is_available())


def MiDaS_init():
    global device
    model_type = "DPT_Large"  # Choose the model type: "DPT_Large", "DPT_Hybrid", or "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    midas.to(device)
    midas.eval()

    # Load transforms
    transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = transform.dpt_transform
    else:
        transform = transform.small_transform

    return midas, transform, device


midas, transform, MiDaS_device = MiDaS_init()


@Decorators.Loggers.log_method_time
def MiDaS_predict(image):
    global midas, transform, device
    input_batch = transform(image).to(device)  # tensor?

    # Predict depth
    with torch.no_grad():
        depth_pred = midas(input_batch)

    depth_pred = torch.nn.functional.interpolate(
        depth_pred.unsqueeze(1),
        size=image.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()
    depth_pred = Helper.Image.normalize_map(depth_pred)
    depth_pred = (depth_pred * 255).astype(np.uint8)
    # save_image("-DepthMap", depth_pred)
    return depth_pred


# img = Image(f"../../../data/input/moon.jpg")()

# result = MiDaS_predict(img)

# Plotter.image(result)
