"""
 (25%) Frequency smoothing:
(a) ComputeFouriertransformofthegivenimagelenaNoise.PNGbyusingnumpy.fft.fft2
in Python, and then center the low frequencies (e.g., by using fftshift).
(b) Keep different number of low frequencies (e.g., 72,152,312 and the full dimension), but
set all other high frequencies to 0.
(c) Reconstruct the original image (ifft2) by using the new generated frequencies in step (b).

"""
# Steps: 
""" 
1. Modify lenaNoise to only require cv2 to procdess images rather than NP. 
2. Organize Mentally Importance behind shifting the 2d fourier transfoorm into the center
    image - what amn I looking at why is ti important? 
    - why are negative signals important? why are positive signals important? df - exploration of the specific signals.
    - phase - how can certain mathematical concepts help to determine what "waves" are present in the function and their phases?
    - real components
3. Implement a backbone implementation of the total variation code
4. Organize a Report Detailing relevant subscriptions.


"""

# Import the image using pil and parse it using a np array.

import cv2
import numpy as np

import matplotlib.pyplot as plt



img = cv2.imread("lenaNoise.png", cv2.IMREAD_GRAYSCALE)


array = np.array(img)
frequencies = np.fft.fft2(img)





# Compute magnitude spectrum (log scale for better visualization)
magnitude_spectrum = np.log(np.abs(frequencies) + 1)

# Apply fftshift

# Why is fftshift important in shifting the freqwuencies
frequencies_shifted = np.fft.fftshift(frequencies)
magnitude_spectrum_shifted = np.log(np.abs(frequencies_shifted) + 1)


# Create mask

mask_size = 30
rows, cols = array.shape
crow, ccol = rows // 2, cols // 2 # This details the center row to be at the center of image.
mask = np.zeros((rows, cols), dtype=np.uint8)
mask[crow-mask_size:crow+mask_size, ccol-mask_size:ccol+mask_size] = 1




# 4. Apply the mask to the shifted Fourier transform
fshift_masked = frequencies_shifted * mask


new_inv_fshift = np.fft.ifftshift(fshift_masked)

# 5. Inverse shift the result
f_ishift = np.fft.ifft2(new_inv_fshift)

# 6. Compute the inverse Fourier transform
image_new = np.abs(f_ishift)



# Create a figure with subplots
fig, axs = plt.subplots(1, 4, figsize=(18, 6))

# Display original image
axs[0].imshow(array, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')




# Display magnitude spectrum without shift
im1 = axs[1].imshow(magnitude_spectrum, cmap='viridis')
axs[1].set_title('Magnitude Spectrum (fft2)')
axs[1].axis('off')
plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

# Display magnitude spectrum with shift
im2 = axs[2].imshow(magnitude_spectrum_shifted, cmap='viridis')
axs[2].set_title('Magnitude Spectrum (fft2 + fftshift)')
axs[2].axis('off')
plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)


axs[3].imshow(image_new, cmap='gray')
axs[3].set_title('Original Image')
axs[3].axis('off')

# Adjust layout and display
plt.tight_layout()
plt.show()



print(array.shape)
print(frequencies.shape)
#print(frequencies)
