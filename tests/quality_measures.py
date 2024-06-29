
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import cv2
from skimage.metrics import structural_similarity as ssim
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


def edge_histograms(image1, image2):
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


def similarity_structure(original_path, retargeted_path, block_size=32):
    # Load images
    original = cv2.imread(original_path)
    retargeted = cv2.imread(retargeted_path)

    # Convert images to grayscale
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    retargeted_gray = cv2.cvtColor(retargeted, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(original_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(retargeted_gray, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Use homography to warp retargeted image
    height, width, channels = original.shape
    retargeted_aligned = cv2.warpPerspective(retargeted, h, (width, height))

    # Calculate local fidelity scores
    h, w, _ = original.shape
    h_r, w_r, _ = retargeted_aligned.shape
    local_fidelity_scores = []

    # # Calculate the block sizes in the retargeted image
    # block_size_h_r = h_r / (h / block_size)
    # block_size_w_r = w_r / (w / block_size)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # Define blocks in the original image
            block_original = original[i:i + block_size, j:j + block_size]

            # Calculate the corresponding block positions in the retargeted image
            start_i_r = int(i / h * h_r)
            end_i_r = int((i + block_size) / h * h_r)
            start_j_r = int(j / w * w_r)
            end_j_r = int((j + block_size) / w * w_r)

            # Define corresponding blocks in the retargeted image
            block_retargeted = retargeted_aligned[start_i_r:end_i_r, start_j_r:end_j_r]

            if block_original.shape[:2] == block_retargeted.shape[:2]:
                # Compute SSIM between the blocks
                score, _ = ssim(block_original, block_retargeted, full=True, channel_axis=2)
                local_fidelity_scores.append(score)

    return np.mean(local_fidelity_scores)


image1_path = (f"{DataPath.INPUT_PATH.value}/bicycle2.png-orginal.png")
image2_path = (f"{DataPath.OUTPUT_PATH.value}/bicycle2.pngSCBuilt-in.png")
content_loss = calculate_content_loss(image1_path, image2_path)
print("Content loss between the two images:", content_loss.numpy())

image1 = Image(f"{DataPath.INPUT_PATH.value}/bicycle2.png-orginal.png")()
image2 = Image(f"{DataPath.OUTPUT_PATH.value}/bicycle2.pngSCBuilt-in.png")()

distance = edge_histograms(image1, image2)
print("edge_histograms between the two images:", distance )

structural_similarity = extract_and_compare_features_SIFT(image1, image2)

psnr_value = compute_psnr(image1_path, image2_path)
print("PSNR between the two images:", psnr_value)

structural_similarity = similarity_structure(image1_path, image2_path)
print(f"Similarity Score: {structural_similarity}")

composite_score =0.5*structural_similarity+ 0.4 * (1/content_loss) + 0.3 * structural_similarity + 0.2 * (1 / distance)+0.1*psnr_value

print("composite score of the image" , composite_score.numpy())

