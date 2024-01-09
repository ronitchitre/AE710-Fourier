import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil

import constants

def make_blocks(g, N, olp):
    olp_frac = olp*0.01
    num_blocks = ceil(g.shape[0] / (N - floor(olp*N)))
    block_array = [A[(i*N - floor(olp_frac*N)*i): (i+1)*N - floor(olp_frac*N)*i] for i in range(num_blocks)]
    return block_array

def windowing(block_array, w):
    windowed_block_arr = [w*block for block in block_array]
    return windowed_block_arr

def DFT(windowed_block_arr, N):
    num_fourier_term = floor((N - 1) / 2)
    k_arr = np.arange(0, N, 1)
    fourier_phases = constants.fourier_phase(k_arr.reshape(-1, 1), np.arange(num_fourier_term), N)
    result = np.dot(windowed_block_arr, fourier_phases)
    return result