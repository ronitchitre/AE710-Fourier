import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil

import constants

def make_blocks(g, N, olp):
    olp_frac = olp*0.01
    num_blocks = floor((g.shape[0] - floor(olp_frac*N)) / (N - 1*floor(olp_frac*N))) 
    block_array = [g[(i*N - floor(olp_frac*N)*i): (i+1)*N - floor(olp_frac*N)*i] for i in range(num_blocks)]
    return block_array

def windowing(block_array, w, scaling):
    windowed_block_arr = w * np.array(block_array) * scaling
    return windowed_block_arr

def DFT(windowed_block_arr, N):
    num_fourier_term = floor((N - 1) / 2)
    k_arr = np.arange(0, N, 1)
    fourier_phases = constants.fourier_phase(k_arr.reshape(-1, 1), np.arange(num_fourier_term), N)
    fourier_terms = np.dot(windowed_block_arr, fourier_phases)
    return fourier_terms

def PSD(fourier_terms, N, fs):
    xeta = N / fs
    half_power = xeta * np.sum(abs(fourier_terms)**2, axis=0) / fourier_terms.shape[0]
    full_psd_scale = 2*np.ones_like(half_power)
    full_psd_scale[0] = 1
    return full_psd_scale * half_power