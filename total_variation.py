
"""
Total Variation Algorithm - An algorithm designed to iteratively denoise an image using gradient descent.
"""


# Create a method for generating noisy images using gaussian added noise.
# Create an algorithm that iteratively denoises an image using total variation.
# Implement * note for any other details of importance in the how part.

import numpy as np
from PIL import Image

def add_gaussian_noise(image, mean=0, sigma=25):
    # Convert PIL image to NumPy array
    img_array = np.array(image)
    

    noise = np.random.normal(mean, sigma, img_array.shape)
    
    noisy_img_array = img_array + noise
    noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)
    
 
    noisy_img = Image.fromarray(noisy_img_array)
    
    return noisy_img