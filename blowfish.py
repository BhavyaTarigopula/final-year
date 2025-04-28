import cv2
import numpy as np
from Crypto.Cipher import Blowfish
from Crypto.Random import get_random_bytes
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from scipy.stats import entropy

# Load image in grayscale
def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Save image
def save_image(image, output_path):
    cv2.imwrite(output_path, image)

# Pad image for Blowfish (multiples of 8 bytes)
def pad_image(image):
    rows, cols = image.shape
    pad_cols = (8 - (cols % 8)) % 8
    padded_image = np.pad(image, ((0, 0), (0, pad_cols)), mode='constant', constant_values=0)
    return padded_image, pad_cols

# Unpad image
def unpad_image(image, pad_cols):
    return image[:, :-pad_cols] if pad_cols > 0 else image

# Blowfish Encryption (CBC Mode)
def blowfish_encrypt(image, key):
    iv = get_random_bytes(8)  # Generate random IV
    cipher = Blowfish.new(key, Blowfish.MODE_CBC, iv)
    padded_image, pad_cols = pad_image(image)

    encrypted_data = cipher.encrypt(padded_image.tobytes())
    encrypted_image = np.frombuffer(encrypted_data, dtype=np.uint8).reshape(padded_image.shape)

    return encrypted_image, pad_cols, iv

# Blowfish Decryption
def blowfish_decrypt(encrypted_image, key, pad_cols, iv):
    cipher = Blowfish.new(key, Blowfish.MODE_CBC, iv)
    decrypted_data = cipher.decrypt(encrypted_image.tobytes())
    decrypted_image = np.frombuffer(decrypted_data, dtype=np.uint8).reshape(encrypted_image.shape)
    return unpad_image(decrypted_image, pad_cols)

# Calculate PSNR, SSIM, Entropy, etc.
def calculate_metrics(original, decrypted, encrypted):
    return {
        "MSE": np.mean((original.astype(np.float32) - decrypted.astype(np.float32)) ** 2),
        "PSNR": psnr(original, decrypted, data_range=255),
        "SSIM": ssim(original, decrypted, data_range=255),
        "Entropy": entropy(np.histogram(encrypted, bins=256, range=(0, 256))[0], base=2)
    }

# Main function
def blowfish_image_encryption_decryption(image_path):
    original_image = load_image(image_path)
    if original_image is None:
        print("Error: Image not found!")
        return

    key = b"12345678"  # Key must be 8-56 bytes
    encrypted_image, pad_cols, iv = blowfish_encrypt(original_image, key)
    decrypted_image = blowfish_decrypt(encrypted_image, key, pad_cols, iv)

    save_image(encrypted_image, "blowfish_encrypted.png")
    save_image(decrypted_image, "blowfish_decrypted.png")

    metrics = calculate_metrics(original_image, decrypted_image, encrypted_image)

    print("Blowfish Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

# Run
blowfish_image_encryption_decryption("cancer.jpg")
