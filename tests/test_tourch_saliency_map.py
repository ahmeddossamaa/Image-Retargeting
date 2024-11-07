import cv2
import numpy as np
from config.constants import DataPath
from config.decorators import Decorators
from src.processors.sc.ConnectedSC import ConnectedSC
from src.processors.SobelFilter import SobelFilter
from config.plotter import Plotter
from scipy.ndimage import sobel
import matplotlib.pyplot as plt
import time
from PIL import Image

import torch
from torchvision import models, transforms

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()

name = "img_5.png"

image_path = f"{DataPath.INPUT_PATH.value}/{name}"



def generate_gradient_map(image):
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Sobel filter in both x and y directions
    sobel_x = sobel(image, axis=0)
    sobel_y = sobel(image, axis=1)

    # Combine the gradient maps
    gradient_map = np.hypot(sobel_x, sobel_y)

    return gradient_map

img = Image.open(image_path)
img = img.convert('RGB')  # Convert the image to RGB if it's not
img_new = img

def preprocess_image(img):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = preprocess(img).unsqueeze(0)
    img_tensor.requires_grad = True
    return img_tensor

def generate_saliency_map(img, model):
    img_tensor = preprocess_image(img)
    img_tensor.requires_grad_()

    logits = model(img_tensor)

    max_index = logits.argmax(dim=1)

    one_hot = torch.zeros_like(logits).scatter_(1, max_index.unsqueeze(-1), 1.0)

    model.zero_grad()
    logits.backward(gradient=one_hot)

    gradients = img_tensor.grad.data.abs().squeeze()

    return gradients

def load_depth_map(depth_map_path):
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)
    return depth_map

@Decorators.Loggers.log_class_method_time
def combine_maps(gradient_map, saliency_map, depth_map):
    # Convert the maps to float32 for normalization
    gradient_map = gradient_map.astype(np.float32)
    saliency_map = saliency_map.astype(np.float32)
    depth_map = depth_map.astype(np.float32)

    # Normalize the maps to have the same range
    gradient_map = cv2.normalize(gradient_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    saliency_map = cv2.normalize(saliency_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    depth_map = cv2.normalize(depth_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Combine the maps
    energy_map = ( saliency_map + depth_map +gradient_map) / 3

    return energy_map

input_image = cv2.imread(image_path)


def resize_map_to_match(map_to_resize, target_shape):
    # Ensure the input is a NumPy array
    map_to_resize = np.array(map_to_resize)
    return cv2.resize(map_to_resize, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

# Measure time
start_time = time.time()

gradient_map = generate_gradient_map(input_image)
saliency_map = generate_saliency_map(img, model)
depth_map = load_depth_map(image_path)

gradient_map_shape = gradient_map.shape
print(gradient_map_shape)
# Resize saliency_map and depth_map to match gradient_map
saliency_map_resized = resize_map_to_match(saliency_map, gradient_map_shape)
depth_map_resized = resize_map_to_match(depth_map, gradient_map_shape)
depth_map_gray = cv2.cvtColor(depth_map_resized, cv2.COLOR_BGR2GRAY)

# energy_map = combine_maps(gradient_map, saliency_map_resized, depth_map_gray)
energy_map= saliency_map


print("test1")
img_np = np.array(img_new)

img_new = ConnectedSC(img_np, energy_map, 0.75, color=False)()


# Calculate time taken
execution_time = time.time() - start_time
print("Execution time: {:.2f} seconds".format(execution_time))

plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(saliency_map_resized, cmap='jet')
plt.title('Saliency Map')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(depth_map_resized, cmap='jet')
plt.title('Resized Depth Map')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(depth_map_gray, cmap='gray')
plt.title('Grayscale Depth Map')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(energy_map, cmap='jet')
plt.title('Energy Map')
plt.axis('off')

plt.show()

Plotter.images([input_image, img_new], 1, 2) 
