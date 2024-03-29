import cv2

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
