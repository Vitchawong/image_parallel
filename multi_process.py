
import os
import multiprocessing
import cv2
import numpy as np
import math
import convolution
import threading
import time
from multiprocessing import Process, shared_memory

if __name__ == "__main__":
    # Read and resize the image
    img = cv2.imread("Resources/mountain_image.jpg", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (500, 500))

    # Add padding to the image
    img = np.pad(img, [(1, 1), (1, 1)], mode='constant', constant_values=0)

    # Display the original image
    cv2.imshow("img", img)
    cv2.waitKey(0)

    print(os.getpid())

    # Initialize the final image array
    final_image = np.zeros(shape=(img.shape[0], img.shape[1]))

    num_processes = 8
    rows_n = int(img.shape[0] / num_processes)

    process_pool = []

    # Define the Laplacian and Gaussian filters
    laplace = [[0, -1, 0],
               [-1, 4, -1],
               [0, -1, 0]]

    gaussian = [[1/16, 2/16, 1/16],
                [2/16, 4/16, 2/16],
                [1/16, 2/16, 1/16]]

    # Create shared memory for the final image
    shm = shared_memory.SharedMemory(
        create=True, size=final_image.nbytes, name="shr_mem")
    final_image = np.ndarray(
        img.shape, dtype=np.float32, buffer=shm.buf)

    for i in range(num_processes):
        # Create a process for each section of the image
        process_pool.append(Process(target=convolution.convolve_multi_process, args=(
            img, i*rows_n, (i+1)*rows_n, 1, img.shape[1]-1, laplace)))

    start = time.time()

    # Start the processes
    for i in process_pool:
        i.start()

    # Wait for the processes to finish
    for i in process_pool:
        i.join()

    end = time.time()
    print("time taken: ", end-start)

    # Display the resulting image
    cv2.imshow("img", final_image/255)
    cv2.waitKey(0)

    # Save the resulting image
    cv2.imwrite("Resources/laplace_edge_output.png", final_image)
