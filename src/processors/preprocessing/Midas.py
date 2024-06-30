import numpy as np
import torch
from config.decorators import Decorators
from config.helper import Helper
from config.plotter import Plotter
from src.processors.Combiner import Combiner
from src.processors.SaliencyMap import SaliencyMap
from util.Image import Image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



print("PyTorch Version: ", torch.__version__)
print("CUDA Version: ", torch.version.cuda)
print("CUDA Available: ", torch.cuda.is_available())
print("CUDA Device Count: ", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current CUDA Device: ", torch.cuda.current_device())
    print("CUDA Device Name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA is not available. Please check your installation.")
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


# img = Image(f"../{DataPath.OUTPUT_PATH.value}/frames/ball.jpg")

# result = MiDaS_predict(img())
# saliency = Combiner(img)().image()
#
# Plotter.images([np.array(saliency)], 1, 2)
