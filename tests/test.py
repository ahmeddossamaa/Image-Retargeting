import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from config.constants import DataPath


name = "img_1.png"


image_path = f"{DataPath.INPUT_PATH.value}/{name}"

model_type = "dpt_swin2_large_384"
model_path = 'C:/Users/GH/Desktop/backend/Image-Retargeting/models/dpt_swin2_large_384.pt'

model = torch.hub.load("intel-isl/MiDaS", model_type)
model.eval()
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Load and transform the image

img = Image.open(image_path)
transform = T.Compose([
    T.Resize(384),
    T.CenterCrop(384),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Predict and save depth map
with torch.no_grad():
    prediction = model(input_tensor)

    # Convert prediction to image and save
    depth_map = prediction.squeeze().cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())  # Normalize
    cv2.imwrite('depth_map.png', depth_map * 255)  # Convert to 8-bit image and save

print("Depth map saved as depth_map.png")