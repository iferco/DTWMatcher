
from utils import *
import librosa
import numpy as np
import os
import pandas as pd 
import pickle

# Duration 15 sr=44100, hop_length=1024, n_bins=84, bins_per_octave=12 70%
#
#Duration: 35 sec 2048 hop length - 969
def compute_cqt(audio_path, sr=44100, hop_length=1024, n_bins=84, bins_per_octave=12, duration=15):
    """
    Params:
        audio_path: Path to the audio file
        sr: Sampling rate
        hop_length: Hop length
        n_bins: Number of frequency bins
        bins_per_octave: Number of bins per octave
        duration: Duration of the audio file for which the CQT is performed.
    Returns:
        cqt: CQT of the audio file
    This function computes and returns the CQT of an audio signal.
    """
    #print("Params" ,sr, hop_length, n_bins, bins_per_octave)
    audio, _ = librosa.load(audio_path, sr=sr, duration=duration)
    cqt = librosa.cqt(audio, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave)
    return cqt



#4 bands, 8 levels
#1938 size - 3 8
def hash_cqt(cqt, num_bands=3, quantization_levels=8):
    """
    Params:
        cqt: CQT of an audio file
        num_bands: Number of frequency bands to divide the CQT into
        quantization_levels: Number of quantization levels for each frequency band
    Returns: 
        cqt_hash: Hashed CQT
    Simple hash algorithm to create a hash for CQT results. It sums the magnitude of each CQT frame and then
    it computes its mean, normalizes it and quantizes it. This results are used to create a hash, which is returned.
    """
    # Calculate the mean magnitude for each frequency band
    band_size = cqt.shape[0] // num_bands
    band_means = [np.mean(np.abs(cqt[i:i+band_size]), axis=0) for i in range(0, cqt.shape[0], band_size)]

    # Normalize each frequency band
    band_means_normalized = [band_mean / np.max(band_mean) for band_mean in band_means]

    # Quantize the normalized magnitudes
    cqt_quantized = [np.floor(band_mean_normalized * quantization_levels).astype(int) for band_mean_normalized in band_means_normalized]

    # Combine the quantized values and create a hash
    cqt_hash = ''.join([''.join(map(str, band_quantized)) for band_quantized in cqt_quantized])
    
    return cqt_hash


def save_hashed_cqt(hashed_cqt, file_id, hashed_cqt_dict):
    """ 
    Params:
        hashed_cqt: Hashed CQT of an audio file
        file_id: ID of the audio file
        hashed_cqt_dict: Dictionary to store the hashed CQTs
    This function stores the hashed CQT of an audio file in a dictionary.
    """
    hashed_cqt_dict[file_id] = hashed_cqt


def process_audio_file(audio_path):
    """
    Process an audio file.
    Params:
        audio_path: Path to the audio file
    This function combines the previous two functions to compute the CQT of an audio file and then hash the CQT.
    """
    cqt = compute_cqt(audio_path)
    hashed_cqt = hash_cqt(cqt)
    return hashed_cqt




from concurrent.futures import ThreadPoolExecutor

def compute_dtw_cost(hashed_cqt, saved_hashed_cqt):
    D, _ = librosa.sequence.dtw(X=np.array(list(hashed_cqt), dtype=float),
                                 Y=np.array(list(saved_hashed_cqt), dtype=float))
    cost = D[-1, -1]
    return cost

def dtw_cost_wrapper(args):
    return compute_dtw_cost(*args)

def find_top_matches(hashed_cqt, hashed_cqt_dict, top_n=115):
    """
    Params:
        hashed_cqt: Hashed CQT of an audio file
        hashed_cqt_dict: Dictionary containing the hashed CQTs of the database
        top_n: Number of top matches to return. 
    Returns:
        top_matches: List of tuples containing the ID of the audio file  (For this case, MusicNet ID) and the DTW cost.
    Finds the top matches after aligning the hashed CQT of an audio file 
    with the hashed CQTs of the database.
    """
    dtw_costs = []

    with ThreadPoolExecutor() as executor:
        dtw_costs = list(executor.map(dtw_cost_wrapper, [(hashed_cqt, cqt) for cqt in hashed_cqt_dict.values()]))

    sorted_dtw_costs = sorted(enumerate(dtw_costs), key=lambda x: x[1])
    top_matches = [(list(hashed_cqt_dict.keys())[idx], cost) for idx, cost in sorted_dtw_costs[:top_n]]
    return top_matches

def create_hashed_cqt_dict(audio_folder):
    """
    Creates a dictionary to store the hashed CQTs of the database.
    Params:
        audio_folder: Path to the folder containing the audio files.
    """
    hashed_cqt_dict = {}
    for file_name in os.listdir(audio_folder):
        if file_name.endswith(".wav"):
            file_path = os.path.join(audio_folder, file_name)
            cqt = compute_cqt(file_path)
            hashed_cqt = hash_cqt(cqt)

            id=file_name.split(".")[0]
            id=id.split("_")[0]
            
            save_hashed_cqt(hashed_cqt, id, hashed_cqt_dict)
    print("Feature dict created")
    return hashed_cqt_dict