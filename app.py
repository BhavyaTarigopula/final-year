from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import streamlit as st
import base64
import numpy as np
import pandas as pd
import cv2
import io
from Crypto.Cipher import Blowfish
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from scipy.stats import entropy

app = FastAPI()

# Encryption methods
class Encryptor:
    def __init__(self, key):
        self.key = key

    def encrypt_blowfish(self, data):
        cipher = Blowfish.new(self.key[:32], Blowfish.MODE_CBC)
        padded_data = self.pad_data(data)
        encrypted_data = cipher.encrypt(padded_data)
        return base64.b64encode(cipher.iv + encrypted_data)

    def decrypt_blowfish(self, encrypted_data):
        encrypted_data = base64.b64decode(encrypted_data)
        iv = encrypted_data[:8]
        cipher = Blowfish.new(self.key[:32], Blowfish.MODE_CBC, iv)
        decrypted_data = cipher.decrypt(encrypted_data[8:])
        return self.unpad_data(decrypted_data)
    
    def encrypt_2dna(self, data):
        return base64.b64encode(data[::-1])  # Placeholder for 2DNA Encoding

    def decrypt_2dna(self, encrypted_data):
        return base64.b64decode(encrypted_data)[::-1]

    def encrypt_noise_crypt(self, data):
        return base64.b64encode(np.roll(np.frombuffer(data, dtype=np.uint8), 3))  # Placeholder for Noise-Crypt

    def decrypt_noise_crypt(self, encrypted_data):
        return np.roll(base64.b64decode(encrypted_data), -3).tobytes()
    
    def encrypt_hybrid(self, data):
        dna_encrypted = self.encrypt_2dna(data)
        return self.encrypt_noise_crypt(dna_encrypted)

    def decrypt_hybrid(self, encrypted_data):
        noise_decrypted = self.decrypt_noise_crypt(encrypted_data)
        return self.decrypt_2dna(noise_decrypted)
    
    def pad_data(self, data):
        block_size = Blowfish.block_size
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length]) * padding_length
        return data + padding
    
    def unpad_data(self, data):
        return data[:-data[-1]]

# Function to compute security metrics
def compute_metrics(original, encrypted):
    original_np = np.array(original.convert('L'))
    encrypted_np = np.array(encrypted.convert('L'))
    
    mse = np.mean((original_np - encrypted_np) ** 2)
    uaci = np.mean(np.abs(original_np.astype(float) - encrypted_np.astype(float)) / 255) * 100
    
    metrics = {
        "PSNR": psnr(original_np, encrypted_np, data_range=255),
        "SSIM": ssim(original_np, encrypted_np, data_range=255),
        "MSE": mse,
        "Entropy": entropy(encrypted_np.flatten()),
        "UACI": uaci,
    }
    return metrics

# Streamlit app
st.title("Secure Medical Image Encryption")
uploaded_file = st.file_uploader("Upload a medical image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    original_image = Image.open(uploaded_file)
    if st.checkbox("View Original Image"):
        st.image(original_image, caption="Original Image", use_container_width=True)
    
    encryptor = Encryptor(b"secure_key_32_bytes_long_here")
    
    encrypted_images = {}
    decrypted_images = {}
    metrics_results = {}
    
    for method in ["Blowfish", "2DNA Encoding", "Noise-Crypt", "Hybrid (2DNA + Noise-Crypt)"]:
        if method == "Blowfish":
            encrypted_data = encryptor.encrypt_blowfish(original_image.tobytes())
            decrypted_data = encryptor.decrypt_blowfish(encrypted_data)
        elif method == "2DNA Encoding":
            encrypted_data = encryptor.encrypt_2dna(original_image.tobytes())
            decrypted_data = encryptor.decrypt_2dna(encrypted_data)
        elif method == "Noise-Crypt":
            encrypted_data = encryptor.encrypt_noise_crypt(original_image.tobytes())
            decrypted_data = encryptor.decrypt_noise_crypt(encrypted_data)
        else:
            encrypted_data = encryptor.encrypt_hybrid(original_image.tobytes())
            decrypted_data = encryptor.decrypt_hybrid(encrypted_data)
        
        decrypted_image = Image.frombytes("L", original_image.size, decrypted_data).convert("RGB")
        
        decrypted_images[method] = decrypted_image
        metrics_results[method] = compute_metrics(original_image, Image.frombytes("L", original_image.size, encrypted_data[:len(original_image.tobytes())]))
    
    st.write("### Security Metrics Comparison")
    st.table(pd.DataFrame(metrics_results).T)
    
    best_method = max(metrics_results, key=lambda m: (metrics_results[m]['Entropy'], -metrics_results[m]['PSNR']))
    st.write(f"### Best Encryption Algorithm: {best_method}")
    
    st.write("### Ideal Security Metric Values")
    st.write("- **PSNR**: Higher is better (Low PSNR indicates strong encryption but poor quality retention)")
    st.write("- **SSIM**: Closer to 1 means better structure retention (Low SSIM means stronger encryption)")
    st.write("- **MSE**: Lower is better (Higher values indicate higher distortion, which is expected in strong encryption)")
    st.write("- **Entropy**: Higher is better (Indicates randomness in encrypted data, essential for security)")
    st.write("- **UACI**: Higher is better (Shows how much pixel intensity has changed, higher is more secure)")
    
    selected_method = st.selectbox("Select Decryption Method", ["Blowfish", "2DNA Encoding", "Noise-Crypt", "Hybrid (2DNA + Noise-Crypt)"])
    
    if st.button("Download Decrypted Image"):
        decrypted_bytes = io.BytesIO()
        decrypted_images[selected_method].save(decrypted_bytes, format='PNG')
        decrypted_bytes = decrypted_bytes.getvalue()
        st.download_button(label="Download Decrypted Image", data=decrypted_bytes, file_name=f"decrypted_{selected_method}.png", mime="image/png")
