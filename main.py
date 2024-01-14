import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil


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
    return np.ones_like(k)
    
# def scaling(window_type, N):
#     k_arr = np.arange(0, N, 1)
#     if window_type == "hann":
#         weight_sum = window_Hann(k_arr, N) * window_Hann(k_arr, N)
#         scale = 1 / (weight_sum / N)
#     elif window_type == "haming":
#         weight_sum = window_Hamming(k_arr, N) * window_Hamming(k_arr, N)
#         scale = 1 / (weight_sum / N)
#     else:
#         weight_sum = window_rectwin(k_arr, N) * window_rectwin(k_arr, N)
#         scale = 1 / (weight_sum / N)
    
def fourier_phase(k, n, N):
    return np.exp(1j*(-2*np.pi*n*k) / N)

def make_blocks(g, N, olp):
    olp_frac = olp*0.01
    num_blocks = floor((g.shape[0] - floor(olp_frac*N)) / (N - 1*floor(olp_frac*N))) 
    block_array = [g[(i*N - floor(olp_frac*N)*i): (i+1)*N - floor(olp_frac*N)*i] for i in range(num_blocks)]
    return block_array

def windowing(block_array, w_arr, scaling):
    windowed_block_arr = w_arr * np.array(block_array) * scaling
    return windowed_block_arr

def DFT(windowed_block_arr, N):
    num_fourier_term = floor((N - 1)/2)
    k_arr = np.arange(0, N, 1)
    fourier_phases = fourier_phase(k_arr.reshape(-1, 1), np.arange(num_fourier_term + 1), N)
    fourier_terms = np.dot(windowed_block_arr, fourier_phases) / N
    return fourier_terms, num_fourier_term

def PSD(fourier_terms, N, fs):
    xeta = N / fs
    half_power =  np.mean(xeta * abs(fourier_terms)**2, axis=0)
    full_psd_scale = 2*np.ones_like(half_power)
    full_psd_scale[0] = 1
    return full_psd_scale * half_power

def calPSD(g, N, fs, w, olp):
    blocked_g = make_blocks(g, N, olp)
    if w == "hann":
        scaling_func = window_Hann
    elif w == "hamming":
        scaling_func = window_Hamming
    elif w == "barlett":
        scaling_func = window_Bartlett
    else:
        scaling_func = window_rectwin
    k_arr = np.arange(0, N, 1)
    w_arr = scaling_func(k_arr, N)
    scaling = (np.sum(w_arr*w_arr) / N)**(-0.5)
    windowed_block = blocked_g * w_arr * scaling

    # power_q = np.mean(np.sum(abs(windowed_block)**2, axis = 1) / N)

    frequency_vals, num_fourier_term = DFT(windowed_block, N)
    F = np.arange(0, (num_fourier_term + 1)*fs/N, fs/N)
    PSDg = PSD(frequency_vals, N, fs)
    return PSDg, F

data = np.loadtxt("pressure.txt")
N = 2**12
fs = 50000
PSDg, F = calPSD(data, N, fs, "hann", 0)

plt.figure(figsize=(8, 6))
plt.plot(F[1:], PSDg[1:])

plt.yscale('log')
plt.xscale('log')
plt.xlim(10**2, 10**4) 
plt.ylim(10**-2, 10**1)

plt.xlabel('Frequency (Hz) - Log Scale')
plt.ylabel('$\mathregular{Pa^{2}Hz^{-1}}$- Log Scale')
plt.title('Power Spectral Density Plot')

plt.show()

k_arr = np.arange(0, N, 1)
w_arr = window_Hann(k_arr, N)
scaling = (np.mean(w_arr*w_arr))**(-0.5)
blocked_data = make_blocks(data, N, 0)
windowed_block_data = windowing(blocked_data, w_arr, scaling)