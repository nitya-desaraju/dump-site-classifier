import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import os
from numba import njit
from scipy.ndimage import generate_binary_structure, iterate_structure
from typing import Tuple, Callable, List


def np_array_to_spec_peak(audio_array):
    """Inputs: an np audio array
       Outputs: saves a PNG of a spectrogram, and returns peaks (row, col) array
    """

    #window -- stride length

    def create_spectrogram(): #creates spectogram and saves it as a png
        SAMPLING_RATE = 44100

        #create the plot
        fig, ax = plt.subplots()

        S, freqs, times, im = ax.specgram(
            audio_array,
            NFFT=4096,
            Fs=SAMPLING_RATE,
            window=mlab.window_hanning,
            noverlap=4096 // 2,
            mode='magnitude',
            scale="dB"
        )
        fig.colorbar(im)

        ax.set_xlabel("Time [seconds]")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("Spectrogram of Recording")
        ax.set_ylim(0, 8000)

        #save it as a PNG
        os.makedirs("spectrogram_images", exist_ok=True)
        fig.savefig("spectrogram_images/spectrogram.png", dpi=150, bbox_inches="tight")

        plt.close(fig)

        return S

    # param: minimum amplitude needed to be considered a peak
    def find_peaks(spectrogram): 

        log_S = np.log(spectrogram).ravel()  
        ind = round(len(log_S) * 0.6)  
        minimum_amplitude = np.partition(log_S, ind)[ind] 

        #neighborhood we will be using to find peaks
        base_structure = generate_binary_structure(2,1)
        neighborhood = iterate_structure(base_structure, 3)

        @njit
        def calculate_peaks(spectogram: np.ndarray, row_offsets: np.ndarray, col_offsets: np.ndarray, min_amplitude: float):
            peaks = []

            for col, row in np.ndindex(*spectrogram.shape[::-1]):
                if spectrogram[row, col] <= min_amplitude: #if < min amplitude, not a peak
                    continue

                for dr, dc in zip(row_offsets, col_offsets):
                    if dr == 0 and dc == 0: #comparing with itself
                        continue

                    if not (0 <= row + dr < spectrogram.shape[0]):
                        # not a peak if outside row boundary
                        continue

                    if not (0 <= col + dc < spectrogram.shape[1]):
                        # not a peak if outside column boundary
                        continue

                    if spectrogram[row, col] < spectrogram[row + dr, col + dc]:
                        # this magnitude is smaller than another one
                        break
                else: #must be a peak
                    peaks.append((row, col))

            return peaks
        
        #takes the neighborhood mask and finds location peak locations
        def get_local_peak_locations():

            #gets all (row, col) where neighborhood is true
            neighborhood_row_indices, neighborhood_col_indices = np.where(neighborhood == True)

            #finds relative distance from center
            row_offsets = neighborhood_row_indices - neighborhood.shape[0] // 2
            col_offsets = neighborhood_col_indices - neighborhood.shape[1] // 2

            #based on the relative positioning of the neighborhood elements, we can calculate peaks
            peaks = calculate_peaks(spectrogram, row_offsets, col_offsets, minimum_amplitude)

            #save a spectrogram with peaks
            fig, ax = plt.subplots()
            S2, freqs, times, im = ax.specgram(
                audio_array,
                NFFT=4096,
                Fs=44100,
                window=mlab.window_hanning,
                noverlap=4096 // 2,
                mode='magnitude',
                scale="dB"
            )
            ax.set_xlabel("Time [seconds]")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_title("Spectrogram with Peaks")
            ax.set_ylim(0, 8000)

            peak_freqs = [freqs[r] for r, c in peaks]
            peak_times = [times[c] for r, c in peaks]
            ax.plot(peak_times, peak_freqs, 'ro', markersize=2, label='Peaks')
            ax.legend()

            fig.colorbar(im)
            fig.savefig("spectrogram_images/spectrogram_with_peaks.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        
            return peaks


        return get_local_peak_locations()
        

    
    S = create_spectrogram()
    return find_peaks(S, 0)



#test
# SAMPLING_RATE = 44100
# DURATION = 2.0 
# FREQUENCY = 440  

# t = np.linspace(0, DURATION, int(SAMPLING_RATE * DURATION), endpoint=False)
# audio_array = 0.5 * np.sin(2 * np.pi * FREQUENCY * t)

# np_array_to_spec_peak(audio_array)