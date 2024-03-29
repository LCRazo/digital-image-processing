import cv2
import numpy as np
from matplotlib import pyplot as plt

# Define a list of NIfTI file paths
file_path = 'lung_ct.jpg'

try:
    # Load the lung CT image as grayscale
    lung_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Display the image
    cv2.imshow('Lung CT Image', lung_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Get size of the image
    height, width = lung_image.shape

    # Report size
    print('Size of the image (M x N): {} x {}'. format(width, height))

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