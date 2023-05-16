
import cv2
import numpy as np
import convolution
import time

# Read and resize the image
img = cv2.imread("Resources/mountain_image.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (500, 500))

# Add padding to the image
img = np.pad(img, [(1, 1), (1, 1)], mode='constant', constant_values=0)

# Display the original image
cv2.imshow("img", img)
cv2.waitKey(0)

# Initialize arrays for y and x gradients
imgy = np.zeros(shape=(img.shape[0], img.shape[1]))
imgx = np.zeros(shape=(img.shape[0], img.shape[1]))

# Define the Laplacian filter
KERNELS = {"Edge Detection": np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]), "Sharpen": np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])}
laplace = KERNELS["Edge Detection"]

# Perform convolution on the image with the Laplacian filter
start = time.time()
final = convolution.convolve(
    img, 1, img.shape[0]-1, 1, img.shape[1]-1, laplace)
end = time.time()

# Print the time taken for convolution
print("time taken: ", end-start)

# Display the resulting image
cv2.imshow("img", final/255)
cv2.waitKey(0)
