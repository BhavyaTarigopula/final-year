import cv2
import numpy as np
import hashlib
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy

# Load the image
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Error: Cannot open or read file {image_path}. Check the file path.")
    return image

# Save the image
def save_image(image, output_path):
    cv2.imwrite(output_path, image)

# Generate SHA-256 hash of the image
def generate_hash(image):
    hash_object = hashlib.sha256(image.tobytes())
    return hash_object.hexdigest()

# Logistic-Tent Map for key generation
def logistic_tent_map(x0, r, iterations):
    sequence = []
    x = x0
    for _ in range(iterations):
        if x < 0.5:
            x = (r * x * (1 - x) + (4 - r) * x / 2) % 1
        else:
            x = (r * x * (1 - x) + (4 - r) * (1 - x) / 2) % 1
        sequence.append(x)
    return sequence

# DNA Encoding
def dna_encode(image):
    return image % 4  # Mapping pixel values directly to 0,1,2,3

# DNA Decoding
def dna_decode(encoded_image):
    return encoded_image  # Direct numerical mapping used for efficiency

# Improved Noise-Crypt with DNA Encoding
def hybrid_encrypt(image, key):
    # Step 1: DNA Encoding
    dna_encoded = dna_encode(image)
    
    # Step 2: Generate chaotic keys using Logistic-Tent Map
    hash_value = generate_hash(image)
    d = int(hash_value[:11], 16)
    x0 = d / (10**14)
    r = 3.99
    key_sequence = logistic_tent_map(x0, r, image.size)
    key_sequence = np.round(np.array(key_sequence) * (10**14)) % 256
    key_sequence = key_sequence.astype(np.uint8).reshape(image.shape)
    
    # Step 3: Perform XOR with chaotic sequence
    encrypted_image = cv2.bitwise_xor(dna_encoded.astype(np.uint8), key_sequence)
    
    return encrypted_image

# Hybrid Decryption
def hybrid_decrypt(encrypted_image, key):
    # Step 1: Generate chaotic key sequence
    hash_value = generate_hash(encrypted_image)
    d = int(hash_value[:11], 16)
    x0 = d / (10**14)
    r = 3.99
    key_sequence = logistic_tent_map(x0, r, encrypted_image.size)
    key_sequence = np.round(np.array(key_sequence) * (10**14)) % 256
    key_sequence = key_sequence.astype(np.uint8).reshape(encrypted_image.shape)
    
    # Step 2: XOR decryption
    decrypted_dna = cv2.bitwise_xor(encrypted_image, key_sequence)
    
    # Step 3: DNA Decoding
    decrypted_image = dna_decode(decrypted_dna)
    
    return decrypted_image

# Performance Metrics
def calculate_psnr(original, decrypted):
    return psnr(original, decrypted, data_range=255)

def calculate_ssim(original, decrypted):
    return ssim(original, decrypted, data_range=255)

def calculate_entropy(image):
    hist = np.histogram(image, bins=256, range=(0, 256))[0]
    hist = hist / hist.sum()
    return entropy(hist, base=2)

def calculate_correlation(original, encrypted):
    return np.corrcoef(original.flatten(), encrypted.flatten())[0, 1]

def calculate_npcr(original, encrypted):
    diff = original != encrypted
    return (np.sum(diff) / original.size) * 100

def calculate_uaci(original, encrypted):
    return np.mean(np.abs(original.astype(float) - encrypted.astype(float)) / 255) * 100

# Main function to run encryption and decryption
def hybrid_image_encryption_decryption(image_path):
    try:
        original_image = load_image(image_path)
        key = 12345  # Example key
        encrypted_image = hybrid_encrypt(original_image, key)
        decrypted_image = hybrid_decrypt(encrypted_image, key)

        # Save results
        save_image(encrypted_image, "hybrid_encrypted.png")
        save_image(decrypted_image, "hybrid_decrypted.png")

        # Calculate metrics
        metrics = {
            "PSNR": calculate_psnr(original_image, decrypted_image),
            "SSIM": calculate_ssim(original_image, decrypted_image),
            "Entropy": calculate_entropy(encrypted_image),
            "Correlation": calculate_correlation(original_image, encrypted_image),
            "NPCR (%)": calculate_npcr(original_image, encrypted_image),
            "UACI (%)": calculate_uaci(original_image, encrypted_image)
        }

        print("Hybrid Encryption Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

    except Exception as e:
        print(f"Error: {e}")

# Run Hybrid Encryption-Decryption
hybrid_image_encryption_decryption("cancer.jpg")
