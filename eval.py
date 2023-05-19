
#Evaluation functions for the matching algorithms.
# Imports: 
import librosa
import numpy as np
import time
from utils import *
from DTWMatcher import *
def get_accuracy(list_match_pos):
    """
    Get accuracy of the matching algorithm. 
    Params:
        list_match_pos: list of positions of the matched compositions
    Returns:
        accuracy: accuracy of the matching algorithm. This means 
        % of the times that the correct composition is in the first position
    """
    accuracy = 0
    for pos in list_match_pos:
        if pos == 0:
            accuracy += 1
    accuracy = accuracy/len(list_match_pos)
    return accuracy*100

def top_10_num_matches(list_match_pos):
    """
    Get the number of times that the correct composition is in the top 10 positions
    Params:
        list_match_pos: list of positions of the matched compositions
    Returns:
        top_10_num_matches: number of times that the correct composition is in the top 10 positions
    """
    top_10_num_matches = 0
    for pos in list_match_pos:
        if pos < 10:
            top_10_num_matches += 1
    return top_10_num_matches

def mean_top_10_pos(list_match_pos):
    """
    Get the mean position of the correct composition in the top 10 positions
    Params:
        list_match_pos: list of positions of the matched compositions
    Returns:
        mean_top_10_pos: mean position of the correct composition in the top 10 positions
    """
    top_10_pos = []
    for pos in list_match_pos:
        if pos < 10:
            top_10_pos.append(pos) 
    return np.mean(top_10_pos)




def ndcg_at_k(positions, k=10):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) at rank k.
    Parameters:
        positions: list of positions of the correct match for different iterations
        k: the rank cutoff
    Returns:
        float, the average NDCG at rank k
    """
    ndcgs = []
    for pos in positions:
        if pos < 0:  # no match found in top k positions
            ndcgs.append(0)
        elif pos < k:  # adjusting for 0-indexing
            dcg = 1 / np.log2(pos + 1 + 1)  # pos + 1 for 0-indexing, and another + 1 because log is undefined at 0
            idcg = 1  # Ideal DCG is 1 because the relevance of the correct match is 1
            #TODO Should idcg be 1 just for items in pos 0? What values to set for the rest?

            ndcgs.append(dcg / idcg)
        else:  # pos >= k
            ndcgs.append(0)
    return np.mean(ndcgs)





