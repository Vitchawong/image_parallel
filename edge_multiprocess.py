import os
from typing import final
import cv2
import numpy as np
import convolution
import time
from multiprocessing import Process, shared_memory

if __name__ == "__main__":
    # FNAME = "blur_img.jpg"
    image_name = "dog.jpg"
    KERNELS = {"Edge Detection": np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]), "Sharpen": np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])}
    kernel = KERNELS["Edge Detection"]
  

    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.resize(image, (500, 500))
    image = np.pad(image, [(1, 1), (1, 1)], 'constant', constant_values=0)

    cv2.imshow("image", image)
    cv2.waitKey(0)
    print(os.getpid())

    final_image = np.zeros(shape=(image.shape[0], image.shape[1]))

    num_processes = 1
    rows_n = int(image.shape[0]/num_processes)

    process_stored = []



    shm = shared_memory.SharedMemory(create=False, size=final_image.nbytes, name="shr_mem")
    final_image = np.ndarray(image.shape, dtype=np.float32, buffer=shm.buf)


    for i in range(num_processes):                                                            #row_start, row_end, col_start, col_end, kernel
        process_stored.append(Process(target=convolution.convolve_multi_process, args=(image, i*rows_n, (i+1)*rows_n, 1, image.shape[1]-1, kernel)))

    start = time.time()
    print(start)
    for i in process_stored:
        i.start()

    for i in process_stored:
        i.join()

    end = time.time()
    print("time taken: ", end-start)

    cv2.imshow("image", final_image/255)
    cv2.waitKey(0)

    cv2.imwrite("FNAME",final_image)
