import numpy as np
from multiprocessing import shared_memory
from numpy.fft import fft2,ifft2


def convolve_multi_process(image=[], row_start=0, row_end=0, col_start=0, col_end=0, kernel=[]):

    existing_shm = shared_memory.SharedMemory(name='shr_mem')

    final_image = np.ndarray(
        (image.shape[0], image.shape[1]), dtype=np.float32, buffer=existing_shm.buf)

    for i in range(row_start, row_end):
        for j in range(col_start, col_end):
            final_image[i][j] = image[i-1][j-1]*kernel[0][0]+image[i-1][j]*kernel[0][1]+image[i-1][j+1]*kernel[0][2]+image[i][j-1]*kernel[1][0] + \
                image[i][j]*kernel[1][1]+image[i][j+1]*kernel[1][2]+image[i+1][j-1] * \
                kernel[2][0]+image[i+1][j]*kernel[2][1] + \
                image[i+1][j+1]*kernel[2][2]
    
