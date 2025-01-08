import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Step 1: Load the image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)  # Convert to grayscale

plt.imshow(image, cmap = 'gray')
plt.show()


# Step 2: Shift using roll image

shifted_image_vertically = np.roll (image, shift = 0, axis = 0) # Shift x px vertically
shifted_image_horizontally = np.roll (shifted_image_vertically, shift = 100, axis = 1) # Shift x px horizontally


# step 3: Cropp image

height, width = np.shape(shifted_image_horizontally)

print (width)
print (height)

top = 0
bottom = height
left = 100
right = width 

 
# Cropped image of above dimension
# (It will not change original image)


shifted_image_horizontally = Image.fromarray(shifted_image_horizontally.astype('uint8'))
cropped_image = shifted_image_horizontally.crop((left, top, right, bottom))

#Display the cropped image as a new file
cropped_image.show()
# Shows the image in image viewer

plt.show()


# step 4: show image


plt.imshow(shifted_image_horizontally, cmap = 'gray')
plt.show()







# Step 2: Apply Fourier Transform
#f_transform = np.fft.fft2(image)                      # 2D Fourier Transform
#f_shift = np.fft.fftshift(f_transform)                # Shift zero frequency to center

# Step 3: Compute amplitude spectrum
#amplitude_spectrum = np.abs(f_shift)

# Step 4: Shift the amplitude spectrum by 1 pixel
#shifted_amplitude = np.roll(amplitude_spectrum, shift=30, axis=0)  # Shift 1 px vertically
#shifted_amplitude = np.roll(amplitude_spectrum, shift=30, axis=1)   # Shift 1 px horizontally

# Step 5: Combine original and shifted amplitude spectrum
#combined_amplitude = amplitude_spectrum * shifted_amplitude  # Combine by multiplication (as in the first code)
#combined_amplitude2 = (amplitude_spectrum*shifted_amplitude)/amplitude_spectrum

# Step 6: Inverse Fourier Transform to return to image domain
# We need to reconstruct the complex Fourier transform, using the combined amplitude spectrum
# Reconstruct the complex spectrum using the combined amplitude and phase of the original Fourier transform
#phase = np.angle(f_shift)  # Phase of the original Fourier transform
#combined_f_transform = combined_amplitude * np.exp(1j * phase)
#combined_f_transform2 = combined_amplitude2 * np.exp(1j * phase)


# Perform inverse Fourier transform to get back to the spatial domain
#reconstructed_image = np.fft.ifft2(np.fft.ifftshift(combined_f_transform))  # Inverse FFT to spatial domain
#reconstructed_image = np.abs(reconstructed_image)  # Take the magnitude (real part)

#reconstructed_image2 = np.fft.ifft2(np.fft.ifftshift(combined_f_transform2))  # Inverse FFT to spatial domain
#reconstructed_image2 = np.abs(reconstructed_image2)  # Take the magnitude (real part)


# Step 7: Visualize results
'''
plt.figure(figsize=(12, 8))

# Original image
plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Original amplitude spectrum
plt.subplot(2, 3, 2)
plt.title("Original Amplitude Spectrum")
plt.imshow(np.log1p(amplitude_spectrum), cmap='gray')
plt.axis('off')

# Shifted amplitude spectrum
plt.subplot(2, 3, 3)
plt.title("Shifted Amplitude Spectrum")
plt.imshow(np.log1p(shifted_amplitude), cmap='gray')
plt.axis('off')

# Combined amplitude spectrum
plt.subplot(2, 3, 3)
plt.title("Combined Amplitude Spectrum")
plt.imshow(np.log1p(combined_amplitude), cmap='gray')
plt.axis('off')

# Combined amplitude spectrum
plt.subplot(2, 3, 4)
plt.title("Combined Amplitude 2 Spectrum")
plt.imshow(np.log1p(combined_amplitude2), cmap='gray')
plt.axis('off')

# Reconstructed image after inverse FFT
plt.subplot(2, 3, 5)
plt.title("Reconstructed Image")
plt.imshow(reconstructed_image, cmap='gray')
plt.axis('off')

# Reconstructed image after inverse FFT
plt.subplot(2, 3, 6)
plt.title("Reconstructed Image2")
plt.imshow(reconstructed_image2, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()'''
