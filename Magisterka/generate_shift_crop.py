import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def generate_shift(image, shift_x, shift_y,):

    # Step 1: Load the image
     # Convert to grayscale

    #plt.imshow(image, cmap = 'gray')
    #plt.show()

    # Step 2: Shift using roll image
    shifted_image = np.roll (image, shift_x, axis = 1) # Shift x px horizontally
    #shifted_image_vertically = np.roll (shifted_image, shift_y, axis = 0) # Shift x px vertically

    # step 4: show image
    plt.imshow(shifted_image, cmap = 'gray')
    plt.show()
    
    return shifted_image

def generate_crop(image, shifted_image, shift_x, shift_y):

        # step 3: Cropp shifted image
    height, width = np.shape(shifted_image)

    #print (width)
    #print (height)

    top = shift_y
    bottom = height
    left = shift_x
    right = width
    height_shift = height-shift_y
    width_shift = width-shift_x
 
 
    image_ = Image.fromarray(image.astype('uint8'))
    cropped_image = image_.crop((left, top, right, bottom))
    
    #Display the cropped image as a new file
    cropped_image.show()
    plt.show()
    
    
    # Cropped image of above dimension
    # (It will not change original image)
    shifted_image_ = Image.fromarray(shifted_image.astype('uint8'))
    cropped_shifted_image = shifted_image_.crop((left, top, right, bottom))

    cropped_shifted_image.show()
    plt.show()

    return cropped_shifted_image, cropped_image, width_shift, height_shift

'''
def hadamard_product(matrix_A, matrix_B):

    #Compute the Hadamard product of two matrices.
    
    Args:
    — matrix_A (numpy.ndarray): First matrix.
    — matrix_B (numpy.ndarray): Second matrix.
    
    Returns:
    — numpy.ndarray: Hadamard product of the input matrices.
    
    # Ensure matrices have the same shape
    if matrix_A.size != matrix_B.size:
        raise ValueError(" Both matrices must have the same shape for Hadamard product.")
 
    #return np.multiply(matrix_A.as_type('float'), matrix_B.as_type('float'))
    return np.multiply(matrix_A, matrix_B)'''

def generate_input(matrix_A, matrix_B, width_shift, height_shift):
    
    print(matrix_A.size)
    print(matrix_B.size)
    
    if matrix_A.size != matrix_B.size:
        raise ValueError(" Both matrices must have the same shape for Hadamard product.")
 
    input_image = np.multiply(matrix_A, matrix_B)
    #print(result)
    #input_image = cropped_image*cropped_shifted_image
    
    plt.imshow(input_image, 'gray')
    plt.show()
    
    # z jakiegoś powodu zwraca inny rozmiar tablicy
    return input_image.reshape((height_shift, width_shift))

def generate_output(matrix_C, matrix_A):
    
    
    matrix_C
    print(matrix_C.size)
    print(matrix_A.size)
    
    #input_image = matrix_C
    if matrix_C.size != matrix_A.size:
        raise ValueError(" Both matrices must have the same shape for Hadamard product.")
    
    #print(result)
    #input_image = cropped_image*cropped_shifted_image
    
    output_image = np.divide(matrix_C, matrix_A)
    
    plt.imshow(output_image, 'gray')
    plt.show()
     
    return output_image


image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

shift_x = 0
shift_y = 0


shifted_image = generate_shift(image,  shift_x, shift_y)

cropped_image, cropped_shifted_image, width_shift, height_shift = generate_crop(image, shifted_image, shift_x, shift_y)

#input_image = hadamard_product(cropped_image, cropped_shifted_image)

input_image = generate_input(cropped_image,cropped_shifted_image, width_shift, height_shift)

plt.imshow(input_image)

plt.show()
print(input_image.size)

output_image = generate_output(input_image,cropped_image)



##niżej raczej nic potrzebnego





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
