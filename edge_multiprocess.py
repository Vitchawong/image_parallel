import os
from typing import final
import cv2
import numpy as np
import convolution
import time
from multiprocessing import Process, shared_memory

if __name__ == "__main__":
    # image_name = "blur_img.jpg"
    image_name = "dog.jpg"
    KERNELS = {"Edge Detection": np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]), "Sharpen": np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])}
    kernel = KERNELS["Edge Detection"]
  
    # make sure the image is black and white
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    # image size
    image = cv2.resize(image, (800, 800))
    print(image)

    # Extend image size (boundary condition handling)
    image = np.pad(image, [(1, 1), (1, 1)], 'constant', constant_values=0)
    padded_img = image
    print(padded_img)


    # display image   
    cv2.imshow("image", image)
    
    # display original image until a key is pressed
    cv2.waitKey(0)


    final_image = np.zeros(shape=(image.shape[0], image.shape[1]))
    num_processes = 6
    rows_n = int(image.shape[0]/num_processes)
    process_stored = []


    # initialize memory
    shm = shared_memory.SharedMemory(create=True, size=final_image.nbytes, name="shr_mem")
    final_image = np.ndarray(image.shape, dtype=np.float32, buffer=shm.buf)


    for i in range(num_processes):                                                            #row_start, row_end, col_start, col_end, kernel
        process_stored.append(Process(target=convolution.convolve_multi_process, args=(image, i*rows_n, (i+1)*rows_n, 1, image.shape[1]-1, kernel)))
        

    start = time.time()
    for i in process_stored:
        i.start()
        

    for i in process_stored:
        i.join()

    end = time.time()
    print("time taken: ", end-start)
    cv2.imshow("image", final_image/255)
    cv2.waitKey(0)
    cv2.imwrite(image_name,final_image)
    
