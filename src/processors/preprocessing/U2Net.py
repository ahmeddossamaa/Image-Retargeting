# from model import U2NET

import cv2
import numpy as np
# from PIL import Image
import torch
from torchvision import transforms

from config.constants import DataPath
from config.helper import Helper
from config.plotter import Plotter
from util.Image import Image


# def U2NET_init():
#     global device
#     model_path = f'../{DataPath.MODELS_PATH.value}/u2net.pth'
#     model = U2NET(3, 1)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device).eval()
#     return model
#
#
# u2net = U2NET_init()


def u2net_predict(image):
    global u2net, device

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    u2net = torch.load(f"../../{DataPath.MODELS_PATH.value}/u2net.pth")
    # u2net.to(device).eval()

    original_image = image.copy()
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = u2net(input)[0].squeeze().cpu().detach().numpy()

    output = (output * 255).astype(np.uint8)
    output = cv2.resize(output, (original_image.width, original_image.height))

    # Add extra boundary
    kernel = np.ones((10, 10), np.uint8)
    dilated = cv2.dilate(output, kernel, iterations=2)

    # Combine original output with dilated boundary
    result = np.maximum(output, dilated)
    result = Helper.Image.normalize_map(result)

    return result


img = Image(f"../../{DataPath.INPUT_PATH.value}/img_4.png")

result = u2net_predict(img.rgb())

Plotter.images([img(), result], 1, 2)
