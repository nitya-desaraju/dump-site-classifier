import pickle
from audio_to_np_array import audio_to_np_array
from np_array_to_multi_array import np_array_to_multi_array
from peaks_to_fingerprints import peaks_to_fingerprints
from database import search
from np_array_to_spec_peak import np_array_to_spec_peak

def recognize_song(filepath: str):
    fan_value = 5 #this value will be tuned for optimization
    
    audio_array = audio_to_np_array(filepath)
    randomly_sampled_arr = np_array_to_multi_array(audio_array)
    peaks = np_array_to_spec_peak(randomly_sampled_arr)
    fingerprints =  peaks_to_fingerprints(peaks, fan_value)

    with open("fingerprints_db.pkl", mode="rb") as opened_file:
      db = pickle.load(opened_file)

    search(fingerprints, db) #no return, this function PRINTS best match song
