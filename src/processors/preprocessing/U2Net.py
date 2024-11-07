from PIL import Image as PILImage
import cv2
import numpy as np
import torch
from torchvision import transforms

from config.constants import DataPath
from config.helper import Helper
from config.plotter import Plotter
from data.models.u2net import U2NET
from util.Image import Image
from util.Processor import Processor

global u2net, device

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_path = f"C:/Users/GH/Desktop/backend/Image-Retargeting/data/models/u2net.pth"
u2net = U2NET(3, 1)  # Initialize the U2NET model
u2net.load_state_dict(torch.load(model_path, map_location=device))
u2net.to(device).eval()


def u2net_predict(image):
    global u2net, device

    original_image = image.copy()
    pil_image = PILImage.fromarray(image)  # Convert NumPy array to PIL image

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = u2net(input)[0].squeeze().cpu().detach().numpy()

    output = (output * 255).astype(np.uint8)

    output = cv2.resize(output, (original_image.shape[1], original_image.shape[0]))  # Corrected dimensions

    # Add extra boundary
    kernel = np.ones((10, 10), np.uint8)
    dilated = cv2.dilate(output, kernel, iterations=2)

    # Combine original output with dilated boundary
    result = np.maximum(output, dilated)
    result = Helper.Image.normalize_map(result)

    return result * 255.0


# img = Image(f"../../{DataPath.INPUT_PATH.value}/img_4.png")
#
# result = u2net_predict(np.array(img.rgb()))  # Convert to NumPy array if not already
#
# Plotter.images([img(), result], 1, 2)
