import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, Sequence
import librosa


def fourier_complex_to_real(
    complex_coeffs: np.ndarray, N: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts complex-valued Fourier coefficients (of 
    real-valued data) to the associated amplitudes and 
    phase-shifts of the real-valued sinusoids
    
    Parameters
    ----------
    complex_coeffs : numpy.ndarray, shape-(N//2 + 1,)
        The complex valued Fourier coefficients for k=0, 1, ...
    
    N : int
        The number of samples that the DFT was performed on.
    
    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        (amplitudes, phase-shifts)
        Two real-valued, shape-(N//2 + 1,) arrays
    """
    amplitudes = np.abs(complex_coeffs) / N

    # |a_k| = 2 |c_k| / N for all k except for
    # k=0 and k=N/2 (only if N is even)
    # where |a_k| = |c_k| / N
    amplitudes[1 : (-1 if N % 2 == 0 else None)] *= 2

    phases = np.arctan2(-complex_coeffs.imag, complex_coeffs.real)
    return amplitudes, phases


def audio_to_np_array(file_path: str) -> np.ndarray:
    audio, sampling_rate = librosa.load(file_path, sr=44100, mono=True)
    time = np.arange(len(audio)) / sampling_rate
    N = len(audio)
    T = N / sampling_rate
    k = np.arange(N // 2 + 1)
    v = k / T #frequencies in Hz

    ck = np.fft.rfft(audio)
    ak, phik = fourier_complex_to_real(ck, N)
    return ak