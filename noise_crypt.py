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

# Logistic-Sine-Cosine Map for random noise generation
def logistic_sine_cosine_map(x0, r, iterations):
    sequence = []
    x = x0
    for _ in range(iterations):
        x = np.cos(np.pi * (4 * r * x * (1 - x) + (1 - r) * np.sin(np.pi * x) - 0.5))
        sequence.append(x)
    return sequence

# Generate S-boxes (AES, Hussain, Gray)
def generate_sboxes():
    # Example S-boxes (replace with actual S-box generation logic)
    aes_sbox = np.random.permutation(256).reshape(16, 16)
    hussain_sbox = np.random.permutation(256).reshape(16, 16)
    gray_sbox = np.random.permutation(256).reshape(16, 16)
    return aes_sbox, hussain_sbox, gray_sbox

# S-box substitution
def sbox_substitution(image, sbox):
    substituted_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            substituted_image[i, j] = sbox[image[i, j] // 16, image[i, j] % 16]
    return substituted_image

# Noise-Crypt Encryption
def noise_crypt_encrypt(image, key):
    # Step 1: Generate hash and derive initial value x0
    hash_value = generate_hash(image)
    d = int(hash_value[:11], 16)  # First 11 characters of hash as decimal
    x0 = d / (10**14)  # Ensure x0 is in [0, 1]

    # Step 2: Generate keys using logistic-tent map
    r = 3.99  # Parameter for logistic-tent map
    key_sequence = logistic_tent_map(x0, r, image.size)
    key_sequence = np.round(np.array(key_sequence) * (10**14)) % 256  # Apply modulo before casting
    key_sequence = key_sequence.astype(np.uint8)  # Now safe to cast to uint8
    key_sequence = key_sequence.reshape(image.shape)  # Ensure shape matches the image

    # Step 3: Perform S-box substitution
    aes_sbox, hussain_sbox, gray_sbox = generate_sboxes()
    substituted_image = sbox_substitution(image, aes_sbox)  # Example: Use AES S-box

    # Step 4: Add random noise using logistic-sine-cosine map
    noise_sequence = logistic_sine_cosine_map(x0, r, image.size)
    noise_sequence = np.round(np.array(noise_sequence) * (10**14)) % 256  # Apply modulo before casting
    noise_sequence = noise_sequence.astype(np.uint8)  # Now safe to cast to uint8
    noise_sequence = noise_sequence.reshape(image.shape)  # Ensure shape matches the image

    # Step 5: Perform Bitwise XOR operations
    encrypted_image = cv2.bitwise_xor(substituted_image, key_sequence)
    encrypted_image = cv2.bitwise_xor(encrypted_image, noise_sequence)

    return encrypted_image

# Noise-Crypt Decryption
def noise_crypt_decrypt(encrypted_image, key):
    # Decryption is the same as encryption due to XOR properties
    return noise_crypt_encrypt(encrypted_image, key)

# Calculate PSNR
def calculate_psnr(original, decrypted):
    return psnr(original, decrypted, data_range=255)


# Calculate MSE
def calculate_mse(original, decrypted):
    return np.mean((original - decrypted) ** 2)

# Calculate SSIM
def calculate_ssim(original, decrypted):
    return ssim(original, decrypted, data_range=255)

# Calculate Entropy
def calculate_entropy(image):
    hist = np.histogram(image, bins=256, range=(0, 256))[0]
    hist = hist / hist.sum()
    return entropy(hist, base=2)

# Calculate Correlation Coefficient
def calculate_correlation(original, encrypted):
    return np.corrcoef(original.flatten(), encrypted.flatten())[0, 1]

# Calculate NPCR (Number of Pixel Change Rate)
def calculate_npcr(original, encrypted):
    diff = original != encrypted
    return (np.sum(diff) / original.size) * 100

# Calculate UACI (Unified Average Changing Intensity)
def calculate_uaci(original, encrypted):
    return np.mean(np.abs(original.astype(float) - encrypted.astype(float)) / 255) * 100

# Main function for Noise-Crypt
def noise_crypt_image_encryption_decryption(image_path):
    try:
        # Load the original image
        original_image = load_image(image_path)

        # Encrypt and decrypt using Noise-Crypt
        key = 12345  # Example key
        encrypted_image = noise_crypt_encrypt(original_image, key)
        decrypted_image = noise_crypt_decrypt(encrypted_image, key)

        # Calculate metrics
        metrics = {
            "PSNR": calculate_psnr(original_image, decrypted_image),
            "SSIM": calculate_ssim(original_image, decrypted_image),
            "Entropy": calculate_entropy(encrypted_image),
            "MSE": calculate_mse(original_image, decrypted_image),
            "Correlation": calculate_correlation(original_image, encrypted_image),
            "NPCR (%)": calculate_npcr(original_image, encrypted_image),
            "UACI (%)": calculate_uaci(original_image, encrypted_image)
        }

        # Print results
        print("Noise-Crypt Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

    except Exception as e:
        print(f"Error: {e}")

# Run Noise-Crypt
noise_crypt_image_encryption_decryption("cancer.jpg")