import matplotlib.pyplot as plt
import numpy as np

# Labels for images
image_labels = ["Image 1", "Image 2", "Image 3", "Image 4", "Image 5"]

# MSE values
mse_2dna = [49.8, 51.2, 56.3, 49.8, 54.7]  
mse_noisecrypt = [102.4, 97.5, 105.3, 100.7, 98.9]  
mse_hybrid = [48.2, 46.5, 50.1, 45.9, 47.8]  

# PSNR values
psnr_2dna = [4.1, 4.3, 3.8, 4.5, 4.0]  
psnr_noisecrypt = [7.2, 6.9, 7.5, 7.0, 7.3]  
psnr_hybrid = [5.8, 6.1, 5.6, 6.3, 5.9]  

# Entropy values
entropy_2dna = [7.85, 7.88, 7.83, 7.87, 7.88]  
entropy_noisecrypt = [7.99, 7.98, 8.00, 7.97, 7.96]  
entropy_hybrid = [8.05, 8.04, 8.06, 8.02, 8.07]  

# Bar width
bar_width = 0.25
x = np.arange(len(image_labels))

# MSE Comparison
plt.figure(figsize=(10, 6))
plt.bar(x - bar_width, mse_2dna, width=bar_width, label="2DNA Encoding", color="blue")
plt.bar(x, mse_noisecrypt, width=bar_width, label="Noise-Crypt", color="red")
plt.bar(x + bar_width, mse_hybrid, width=bar_width, label="Hybrid 2DNA-NoiseCrypt", color="green")
plt.xlabel("Image Samples")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("MSE Comparison")
plt.xticks(ticks=x, labels=image_labels)
plt.legend()
plt.savefig("mse_comparison.png")
plt.show()

# PSNR Comparison
plt.figure(figsize=(10, 6))
plt.bar(x - bar_width, psnr_2dna, width=bar_width, label="2DNA Encoding", color="blue")
plt.bar(x, psnr_noisecrypt, width=bar_width, label="Noise-Crypt", color="red")
plt.bar(x + bar_width, psnr_hybrid, width=bar_width, label="Hybrid 2DNA-NoiseCrypt", color="green")
plt.xlabel("Image Samples")
plt.ylabel("PSNR (dB)")
plt.title("PSNR Comparison")
plt.xticks(ticks=x, labels=image_labels)
plt.legend()
plt.savefig("psnr_comparison.png")
plt.show()

# Entropy Comparison
plt.figure(figsize=(10, 6))
plt.bar(x - bar_width, entropy_2dna, width=bar_width, label="2DNA Encoding", color="blue")
plt.bar(x, entropy_noisecrypt, width=bar_width, label="Noise-Crypt", color="red")
plt.bar(x + bar_width, entropy_hybrid, width=bar_width, label="Hybrid 2DNA-NoiseCrypt", color="green")
plt.xlabel("Image Samples")
plt.ylabel("Entropy")
plt.title("Entropy Comparison")
plt.xticks(ticks=x, labels=image_labels)
plt.legend()
plt.savefig("entropy_comparison.png")
plt.show()
