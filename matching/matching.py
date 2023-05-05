import sys
import os
import pandas as pd
import numpy as np
import librosa, pretty_midi
import pandas as pd
import pretty_midi
import pickle
import time
import multiprocessing
import concurrent.futures as cf
import random
from utils import *
#FUNCTIONS



##################################
#           ALIGNMENT            #


def align_chroma(score_midi, perf, fs=44100, stride=512, n_fft=4096,duration=15):
    """
    DTW Alignment using chroma features:
    Params:
        score_midi: path to score midi file
        perf: path to performance audio file
        fs: sampling rate
        stride: hop length
        n_fft: number of fft bins
        duration: duration of the audio file for which the chroma is performed

    Returns:
        cost: DTW cost of alignment
    """
    score_synth,_=librosa.load(score_midi, sr=fs)
    score_synth = score_synth[:fs*duration]
    #if duration less than 0 or greater than the length of the performance, use the whole performance
    perf,_ = librosa.load(perf, sr=fs)
    perf=perf[:fs*duration]


    
    
    score_chroma = librosa.feature.chroma_stft(y=score_synth, sr=fs, tuning=0, norm=2,
                                               hop_length=stride, n_fft=n_fft)
    score_logch = librosa.power_to_db(score_chroma, ref=score_chroma.max())
    perf_chroma = librosa.feature.chroma_stft(y=perf, sr=fs, tuning=0, norm=2,
                                              hop_length=stride, n_fft=n_fft)
    perf_logch = librosa.power_to_db(perf_chroma, ref=perf_chroma.max())


    #length = length of score_logch/perf_logch, the higher one
    length = max(score_logch.shape[1],perf_logch.shape[1])
    #D is the DTW matrix, wp is the warping path
    D, wp = librosa.sequence.dtw(X=score_logch, Y=perf_logch)
    #cost is the DTW cost of alignment
    cost=D[D.shape[0]-1, D.shape[1]-1]
    #B is the maximum distance between the two features used in the matrix
    #with this value and the length of the performance/score, we can normalize the cost
    cost=dtw_score_norm_l2(cost, perf_logch, score_logch)
    #cost=normalize_cost_2(cost,length)

    return cost


def align_spectral(score_midi, perf, fs=44100, stride=512, n_fft=4096,duration=15):
    """
    DTW Alignment using spectral features:
    Params:
        score_midi: path to score midi file
        perf: path to performance audio file
        fs: sampling rate
        stride: hop length
        n_fft: number of fft bins

    Returns:
        cost: DTW cost of alignment

    """
    score_synth,_=librosa.load(score_midi, sr=fs)
    score_synth = score_synth[:fs*duration]
    #if duration less than 0 or greater than the length of the performance, use the whole performance
    perf,_ = librosa.load(perf, sr=fs)
    perf=perf[:fs*duration]
 
    score_spec = np.abs(librosa.stft(y=score_synth, hop_length=stride, n_fft=n_fft))**2
    score_logspec = librosa.power_to_db(score_spec, ref=score_spec.max())
    perf_spec = np.abs(librosa.stft(y=perf, hop_length=stride, n_fft=n_fft))**2
    perf_logspec = librosa.power_to_db(perf_spec, ref=perf_spec.max())

    #length = length of score_logch/perf_logch, the higher one
    length = max(score_logspec.shape[1],perf_logspec.shape[1])
    #D is the DTW matrix, wp is the warping path
    D, wp = librosa.sequence.dtw(X=score_logspec, Y=perf_logspec)
    #cost is the DTW cost of alignment
    cost=D[D.shape[0]-1, D.shape[1]-1]
    #B is the maximum distance between the two features used in the matrix
    #with this value and the length of the performance/score, we can normalize the cost
    cost=dtw_score_norm_l2(cost, perf_logspec, score_logspec)
    #cost=normalize_cost_2(cost,length)

    return cost


def max_distance(dtw_matrix):
    """
    Params:
        dtw_matrix: DTW matrix
    Given a non-square distance DTW matrix, returns the maximum distance between the two features used in the matrix.
    """
    n = dtw_matrix.shape[0]
    m = dtw_matrix.shape[1]
    max_val = -np.inf
    
    for i in range(n):
        for j in range(m):
            if dtw_matrix[i,j] > max_val:
                max_val = dtw_matrix[i,j]
    
    return max_val


def normalize_cost(cost,length,B):
    """
    Params:
        cost: DTW cost of alignment
        length: length of the performance/score, whichever is longer
    Returns:
        normalized cost
    

    """
    if B == 0 or length == 0 or cost != cost:
        # Return a large value if B or length are zero, or if cost is NaN
        return float('inf')
    else:
        # Compute the cost normally if B and length are non-zero, and cost is a valid number
        return cost/(length*B)


# Second normalization option, suggested in OPTIMIZING DTW-BASED AUDIO-TO-MIDI ALIGNMENT AND MATCHING
#it uses L2 norm
def dtw_score_norm_l2(dtw_score, x, y):
    """
    Params:
        dtw_score: DTW cost of alignment
        x: feature 1
        y: feature 2
    Returns:
            normalized cost
    Normalization of the DTW cost using the L2 norm of the two features used in the alignment.
    Suggested in OPTIMIZING DTW-BASED AUDIO-TO-MIDI ALIGNMENT AND MATCHING by C Raffel et al.
    """
    norm_factor = np.sqrt(np.sum(np.square(x)) + np.sum(np.square(y)))
    return dtw_score / norm_factor




##################################


##################################
#           MATCHING             #
#Do this with all midi files

import concurrent.futures

def process_score(score_path, performance_path, duration):
    alignment_cost = align_chroma(score_path, performance_path, n_fft=4096, duration=duration)
    name = get_full_name(score_path)
    return (name, alignment_cost)

def matching(midi_files, performance_path, duration):
    """
    Params:
        midi_files: list of midi files
        performance_path: path to performance audio file
        duration: duration of the audio file
    Returns:
            scosts: list of tuples (midi_file_name, cost) ordered by cost (ascending)
    The alignment is done using the chroma features of the midi files and the performance 
    to get the DTW cost of alignment. The cost is then normalized and returned .
    """
    costs = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = [executor.submit(process_score, score_path, performance_path, duration) for score_path in midi_files]
        for future in concurrent.futures.as_completed(results):
            costs.append(future.result())
    
    #s_costs is a list of name, cost ordered by cost (ascending)
    s_costs = sorted(costs, key=lambda x: x[1])
    return s_costs


def matching_spectral(midi_files, performance_path,duration):
    """
    Params:
        midi_files: list of midi files
        performance_path: path to performance audio file
        duration: duration of the audio file
    Returns:
            position: position of the performance (wanted track) in the list of midi files.
            -1 if not found, 0 if first, 1 if second, etc.
    The alignment is done using the chroma features of the midi files and the performance 
    to get the DTW cost of alignment. The cost is then normalized and returned .
    """
    #print('Matching...')
    costs=[]
    for score_path in midi_files:
        alignment_cost = align_spectral(score_path,performance_path,n_fft=4092,duration=duration)
        name=get_full_name(score_path)
        costs.append((alignment_cost, name))
        

    name=performance_path.split('/')[-1].split('.')[0]

    wanted_composition=get_full_name(performance_path)
    #print('Wanted score: ', wanted_composition[0], 'Movement: ', wanted_composition[1] )

    #print('Top 10 matches: ')
    #ordered, let's print top 3
    count=0
    s_costs=sorted(costs)
    s_costs=[x[1] for x in s_costs]
    return get_position(s_costs, wanted_composition[0], wanted_composition[1])





##################################
 

 







 