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
    
def fourier_phase(k, n, N):
    return np.exp(1j*(-2*np.pi*n*k) / N)