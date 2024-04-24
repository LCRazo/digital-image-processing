import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage as sci
from sklearn.metrics import mean_squared_error

# Define a list of NIfTI file paths
file_path = 'lung_ct.jpg'

try:
    # Load the lung CT image as grayscale
    lung_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Get size of the image
    height, width = lung_image.shape

    # Display the image
    plt.imshow(lung_image, cmap='gray')
    plt.title('Lung CT Image'), plt.xticks([]), plt.yticks([])
    plt.text(0, height + 20, f'Size of the image (M x N): {width} x {height}')
    plt.show()

except FileNotFoundError:
    print("File not found: {file_path}")

except Exception as e:
    print(f"An error occurred for {file_path}: {e}")


# creating the magnitude spectrum
f = np.fft.fft2(lung_image)
magnitude = 20*np.log(np.abs(f))

# adding a phase shift
phase = np.angle(f)

# applying shift
fshift = np.fft.fftshift(f)
mag_shift = 20*np.log(np.abs(fshift))
phase_shift = np.angle(fshift)

# display the magnitude before and after
plt.subplot(121),plt.imshow(magnitude, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(mag_shift, cmap = 'gray')
plt.title("Shifted Magnitude Spectrum"), plt.xticks([]), plt.yticks([])
plt.show()

# Displaying the phase before and after
x, axarr = plt.subplots(1, 2)
axarr[0].imshow(phase, cmap = 'gray')
axarr[1].imshow(phase_shift, cmap = 'gray')
axarr[0].set_title("Phase")
axarr[1].set_title("Shifted Phase")
plt.show()


# Down-sample image by selecting every other row and column by a factor of 2
scale_factor = 2
downsampled_image = lung_image[::scale_factor, ::scale_factor]


# Display the down-sampled image before and after
plt.subplot(121), plt.imshow(lung_image, cmap = 'gray')
plt.title('Original CT Scan')
plt.subplot(122), plt.imshow(downsampled_image, cmap = 'gray')
plt.title('Down-sampled Image')
plt.axis('on')
plt.show()

# Get frequency domain for down sample image
fft_downsampled_image = np.fft.fft2(downsampled_image)

originalheight, originalwidth = lung_image.shape
newheight, newwidth = fft_downsampled_image.shape
print(newheight, newwidth)
padding_amount_x = (originalwidth - newwidth)
padding_amount_y = (originalheight - newheight)

# Zero-pad FFT downsample image to original size
padded_fft = np.pad(fft_downsampled_image, ((0, padding_amount_y), (0, padding_amount_x)), mode='constant')

# Compute the inverse 2D FFT
inverse_image = np.fft.ifft2(padded_fft)
inverse_image = np.abs(inverse_image)

inverse_h, inverse_w = inverse_image.shape

# Display the interpolated image
plt.imshow(inverse_image, cmap='grey')
plt.title('Magnitude of Reconstructed Image'), plt.xticks([]), plt.yticks([])
plt.text(0, inverse_h + 20, f'Size of the image (M x N): {inverse_w} x {inverse_h}')
plt.show()

# Display the real part of the reconstructed image
plt.imshow(np.real(inverse_image), cmap='grey')
plt.title('Real Part of Reconstructed Image'), plt.xticks([]), plt.yticks([])
plt.show()

# Interpolate image from part 3
interpolated_image = sci.zoom(downsampled_image, scale_factor, order=1)
int_h, int_w = interpolated_image.shape

plt.imshow(np.abs(interpolated_image), cmap='grey')
plt.title('Interpolated Image'), plt.xticks([]), plt.yticks([])
plt.text(0, int_h + 20, f'Size of the image (M x N): {int_w} x {int_h}')
plt.show()


mse_1 = mean_squared_error(lung_image, inverse_image)
mse_2 = mean_squared_error(lung_image, interpolated_image)
print("MSE 1: " + str(mse_1) + " MSE 2: " + str(mse_2))

