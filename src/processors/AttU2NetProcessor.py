import cv2
import numpy as np
import torch
from PIL import Image as PILImage

from config.helper import Helper
from src.processors.preprocessing.AttU2Net import AttU2Net
from util.Image import Image
from util.Processor import Processor


class AttU2NetProcessor(Processor):
    def __init__(self, image: Image, model_path=None):
        super().__init__(image)
        self.model = AttU2Net().to(self.device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            # Initialize with random weights (you'd want to load pretrained weights)
            print("Warning: Using randomly initialized weights. Load pretrained model for better results.")

        self.model.eval()

    def preprocess_image(self, image):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Resize to model input size
        pil_image = PILImage.fromarray(image)
        pil_image = pil_image.resize((320, 320))

        # Convert to tensor
        image_tensor = torch.from_numpy(np.array(pil_image)).float()
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.permute(2, 0, 1)

        # Normalize
        image_tensor = image_tensor / 255.0
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor.to(self.device)

    def main(self, *args, **kwargs):
        image_array = self._image()
        original_size = image_array.shape[:2]

        # Preprocess
        input_tensor = self.preprocess_image(image_array)

        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            saliency = output.squeeze().cpu().numpy()

        # Resize back to original size
        saliency_resized = cv2.resize(saliency, (original_size[1], original_size[0]))
        saliency_normalized = Helper.Math.normalize(saliency_resized, 0, 1)

        self._image = saliency_normalized