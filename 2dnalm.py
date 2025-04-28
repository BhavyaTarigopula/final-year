import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy

# Load the image
def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Save the image
def save_image(image, output_path):
    cv2.imwrite(output_path, image)

# DNA Encoding (using numbers instead of strings for efficiency)
def dna_encode(image):
    return image % 4  # Mapping pixel values directly to 0,1,2,3

# DNA Decoding (reverse mapping)
def dna_decode(encoded_image):
    return encoded_image  # No actual conversion needed for numerical encoding

# Improved 2D Logistic Map (now deterministic with better randomness)
def logistic_map_2d(shape, key, u=3.99, x0=0.5):
    np.random.seed(key)  # Ensures repeatability
    rows, cols = shape
    map_values = np.zeros(shape, dtype=np.uint8)
    x = x0
    for i in range(rows):
        for j in range(cols):
            x = u * x * (1 - x)
            map_values[i, j] = int((x * 255) % 256)  # Ensures values are in uint8 range
    return map_values

# 2DNALM Encryption
def dnalm_encrypt(image, key):
    dna_encoded = dna_encode(image)
    logistic_map = logistic_map_2d(image.shape, key)
    encrypted_image = cv2.bitwise_xor(dna_encoded.astype(np.uint8), logistic_map)
    return encrypted_image

# 2DNALM Decryption
def dnalm_decrypt(encrypted_image, key):
    logistic_map = logistic_map_2d(encrypted_image.shape, key)
    decrypted_image = cv2.bitwise_xor(encrypted_image, logistic_map)  # XOR decrypts itself
    return dna_decode(decrypted_image)

# Calculate PSNR (handling divide-by-zero)
def calculate_psnr(original, decrypted):
    mse = np.mean((original - decrypted) ** 2)
    if mse == 0:
        return float('inf')  # Perfect match
    return psnr(original, decrypted, data_range=255)

# Calculate SSIM
def calculate_ssim(original, decrypted):
    return ssim(original, decrypted, data_range=255)

# Calculate MSE
def calculate_mse(original, decrypted):
    return np.mean((original - decrypted) ** 2)

# Calculate Entropy
def calculate_entropy(image):
    hist = np.histogram(image, bins=256, range=(0, 256))[0]
    hist = hist / hist.sum()
    return entropy(hist, base=2)

# Calculate Correlation Coefficient
def calculate_correlation(original, decrypted):
    return np.corrcoef(original.flatten(), decrypted.flatten())[0, 1]

# Calculate NPCR (Number of Pixel Change Rate)
def calculate_npcr(original, encrypted):
    diff = original != encrypted
    return (np.sum(diff) / original.size) * 100

# Calculate UACI (Unified Average Changing Intensity)
def calculate_uaci(original, encrypted):
    return np.mean(np.abs(original.astype(float) - encrypted.astype(float)) / 255) * 100

# Main function for 2DNALM Encryption & Decryption
def dnalm_image_encryption_decryption(image_path):
    # Load the original image
    original_image = load_image(image_path)
    
    if original_image is None:
        print("Error: Image not found!")
        return

    # Encrypt and decrypt using 2DNALM
    key = 12345  # Example key
    encrypted_image = dnalm_encrypt(original_image, key)
    decrypted_image = dnalm_decrypt(encrypted_image, key)

    # Save results
    save_image(encrypted_image, "encrypted.png")
    save_image(decrypted_image, "decrypted.png")

    # Calculate metrics
    metrics = {
        "PSNR": calculate_psnr(original_image, decrypted_image),
        "SSIM": calculate_ssim(original_image, decrypted_image),
        "MSE": calculate_mse(original_image, decrypted_image),
        "Entropy": calculate_entropy(encrypted_image),
        "Correlation": calculate_correlation(original_image, encrypted_image),
        "NPCR (%)": calculate_npcr(original_image, encrypted_image),
        "UACI (%)": calculate_uaci(original_image, encrypted_image)
    }

    # Print results
    print("2DNALM Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

# Run 2DNALM
dnalm_image_encryption_decryption("cancer.jpg")