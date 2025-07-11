import numpy as np
import random

#helper function called inside of [np_array_to_multi_array]
def random_sample(audio: np.ndarray, time: float, sampling_rate: int = 44100) -> np.ndarray:
    clip_len = int(sampling_rate * time)
    if clip_len >= len(audio):
        raise ValueError("clip length must be less than audio length")
    
    start = np.random.randint(0, len(audio) - clip_len)
    end = start + clip_len
    return audio[start:end]

def np_array_to_multi_array(arr: np.ndarray, num: int, minT: float = 2.0, maxT: float = 10.0, sampling_rate: int = 44100) -> list:
    ret = [] #returns a list of np.ndarrays
    for i in range(num):
        max_time = len(arr) / sampling_rate
        clip_time = random.uniform(minT, min(max_time, maxT))  # Avoid time=0; clips will be between 2 and 10 seconds by default
        ret.append(random_sample(arr, clip_time, sampling_rate))
    return ret
