import numpy as np

def window_Hann(k, N):
    return 0.5*(1 - np.cos((2*np.pi*k)/(N-1)))

def window_Hamming(k, N):
    return 0.54 - 0.46*np.cos((2*np.pi*k)/(N-1))

def window_Bartlett(k, N):
    if k <= (N-1) / 2:
        return 2*k / (N - 1)
    else:
        return 2 - (2*k / (N-1))

def window_rectwin(k, N):
    return 1
    
def scaling(window_type, N):
    k_arr = np.arange(0, N, 1)
    if window_type == "hann":
        weight_sum = window_Hann(k_arr, N) * window_Hann(k_arr, N)
        scale = 1 / (weight_sum / N)
    elif window_type == "haming":
        weight_sum = window_Hamming(k_arr, N) * window_Hamming(k_arr, N)
        scale = 1 / (weight_sum / N)
    else:
        weight_sum = window_rectwin(k_arr, N) * window_rectwin(k_arr, N)
        scale = 1 / (weight_sum / N)
    
def fourier_phase(k, n, N):
    return np.exp(1j*(-2*np.pi*n*k) / N)