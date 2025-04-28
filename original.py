import cv2
import numpy as np
from scipy.stats import entropy, pearsonr
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# Load the image in grayscale
def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Calculate Mean Squared Error (MSE)
def calculate_mse(image):
    return np.mean((image.astype(np.float32) - image.astype(np.float32)) ** 2)  # MSE is 0 for original

# Calculate Peak Signal-to-Noise Ratio (PSNR)
def calculate_psnr(image):
    mse = calculate_mse(image)
    if mse == 0:
        return float('inf')  # Perfect image
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# Calculate Entropy
def calculate_entropy(image):
    pixel_values = image.flatten()
    hist, _ = np.histogram(pixel_values, bins=256, range=(0, 256), density=True)
    return entropy(hist, base=2)  # Base-2 entropy

# Calculate Correlation (Pearson's coefficient)
def calculate_correlation(image):
    left_pixels = image[:, :-1].flatten()  # Excluding last column
    right_pixels = image[:, 1:].flatten()  # Excluding first column
    corr, _ = pearsonr(left_pixels, right_pixels)  # Pearson correlation coefficient
    return corr

# Calculate Structural Similarity Index (SSIM)
def calculate_ssim(image):
    return ssim(image, image)  # SSIM is 1 for the original image

# Path to original image (replace with actual path)
original_image_path = "cancer.jpg"  
original_image = load_image(original_image_path)

# Compute metrics
mse_value = calculate_mse(original_image)
psnr_value = calculate_psnr(original_image)
entropy_value = calculate_entropy(original_image)
correlation_value = calculate_correlation(original_image)
ssim_value = calculate_ssim(original_image)

# Print results
print("Metrics for the Original Image:")
print(f"Mean Squared Error (MSE): {mse_value}")
print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr_value}")
print(f"Entropy: {entropy_value}")
print(f"Correlation: {correlation_value}")
print(f"Structural Similarity Index (SSIM): {ssim_value}")
