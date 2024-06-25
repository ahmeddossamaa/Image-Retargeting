
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from skimage import io
import os
from utils.Image import Image
from config.constants import DataPath

def calculate_content_loss(image_path1, image_path2):
    # Define VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_conv2').output)

    # Load and preprocess images
    image1 = tf.keras.preprocessing.image.load_img(image_path1, target_size=(224, 224))
    image2 = tf.keras.preprocessing.image.load_img(image_path2, target_size=(224, 224))
    image1 = tf.keras.preprocessing.image.img_to_array(image1)
    image2 = tf.keras.preprocessing.image.img_to_array(image2)
    image1 = np.expand_dims(image1, axis=0)
    image2 = np.expand_dims(image2, axis=0)
    image1 = preprocess_input(image1)
    image2 = preprocess_input(image2)

    # Extract features
    features1 = model.predict(image1)
    features2 = model.predict(image2)

    # Calculate MSE loss
    mse_loss = tf.keras.losses.MeanSquaredError()(features1, features2).numpy()
    num_elements = tf.cast(tf.reduce_prod(tf.shape(features1)), dtype=tf.float32)

    content_loss = mse_loss / num_elements

    return content_loss


def compute_and_compare_edge_histograms(image1, image2):
    # Compute edge histogram for the first image
    edges1 = cv2.Canny(image1, 100, 200)
    dx1 = cv2.Sobel(edges1, cv2.CV_64F, 1, 0, ksize=3)
    dy1 = cv2.Sobel(edges1, cv2.CV_64F, 0, 1, ksize=3)
    orientations1 = np.arctan2(dy1, dx1)
    magnitudes1 = np.sqrt(dx1 ** 2 + dy1 ** 2)
    bins = np.linspace(-np.pi, np.pi, 10)
    hist1, _ = np.histogram(orientations1, bins=bins, weights=magnitudes1)
    hist1 /= np.sum(hist1)

    # Compute edge histogram for the second image
    edges2 = cv2.Canny(image2, 100, 200)
    dx2 = cv2.Sobel(edges2, cv2.CV_64F, 1, 0, ksize=3)
    dy2 = cv2.Sobel(edges2, cv2.CV_64F, 0, 1, ksize=3)
    orientations2 = np.arctan2(dy2, dx2)
    magnitudes2 = np.sqrt(dx2 ** 2 + dy2 ** 2)
    hist2, _ = np.histogram(orientations2, bins=bins, weights=magnitudes2)
    hist2 /= np.sum(hist2)

    # Compute distance between histograms
    distance = np.linalg.norm(hist1 - hist2)

    return distance




def extract_and_compare_features_SIFT(image1, image2):
    # Read images

    sift1 = cv2.SIFT().create()
    kp1, des1 = sift1.detectAndCompute(image1, None)
    sift2 = cv2.SIFT().create()
    kp2, des2 = sift2.detectAndCompute(image2, None)

    # Match features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    a = len(good)
    print(a)
    percent = (a * 100) / min(len(kp1), len(kp2))
    print("{} % similarity".format(percent))
    if percent >= 75.00:
        print('Match Found')
    img3 = cv2.drawMatchesKnn(image1, kp1, image2, kp2, matches[:20], None, flags=2)
    return percent
    #plt.imshow(img3), plt.show()

def compute_psnr(image1_path, image2_path):
    # Load the images
    image1 = io.imread(image1_path)
    image2 = io.imread(image2_path)

    # Ensure images have the same shape
    min_shape = np.min([image1.shape, image2.shape], axis=0)
    image1 = image1[:min_shape[0], :min_shape[1]]
    image2 = image2[:min_shape[0], :min_shape[1]]

    # Compute PSNR
    psnr_value = peak_signal_noise_ratio(image1, image2)

    return psnr_value


image1_path = (f"{DataPath.INPUT_PATH.value}/bicycle2.png-orginal.png")
image2_path = (f"{DataPath.OUTPUT_PATH.value}/bicycle2.pngSCBuilt-in.png")
content_loss = calculate_content_loss(image1_path, image2_path)
print("Content loss between the two images:", content_loss.numpy())

image1 = Image(f"{DataPath.INPUT_PATH.value}/bicycle2.png-orginal.png")()
image2 = Image(f"{DataPath.OUTPUT_PATH.value}/bicycle2.pngSCBuilt-in.png")()

distance = compute_and_compare_edge_histograms(image1, image2)
print("edge_histograms between the two images:", distance )

similarity_score = extract_and_compare_features_SIFT(image1, image2)

psnr_value = compute_psnr(image1_path, image2_path)
print("PSNR between the two images:", psnr_value)

# print("similarity_score", similarity_score)

composite_score = 0.4*(1/content_loss) + 0.3 * similarity_score + 0.2 * (1/distance)

print("composite score of the image" , composite_score.numpy())
