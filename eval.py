
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


def calculate_p_at_10(match_positions, iterations=100):
    """
    Calculate Precision at rank 10 (P@10).
    Parameters:
        match_positions: list of positions of the correct match for different iterations
    Returns:
        float, the average P@10
    """
    p_at_10 = 0
    for match in match_positions:
        if match < 10:  # Count matches that occurred in top-10
            p_at_10 += 1

    # Normalize P@10 by the total number of cases we checked (up to 10)
    p_at_10 /= iterations  # 330 is the total number of cases we checked

    return p_at_10


def calculate_mrr(match_positions):
    """
    Calculate Mean Reciprocal Rank (MRR).
    Parameters:
        match_positions: list of positions of the correct match for different iterations
        Returns:
        float, the average MRR
    """
    # Initialize sum of reciprocal ranks
    sum_reciprocal_ranks = 0

    # Count the number of experiments where a match was found
    count_valid_experiments = 0

    for match in match_positions:
        if match is not None and match >= 0:  # Ignore experiments where no match was found
            sum_reciprocal_ranks += 1 / (match + 1)  # add 1 to match because index starts from 0
            count_valid_experiments += 1

    if count_valid_experiments == 0:
        return None  # or whatever you want to return in this case

    # Calculate MRR by taking the average of reciprocal ranks
    mrr = sum_reciprocal_ranks / count_valid_experiments

    return mrr




def calculate_mr1(match_positions):
    """
    Calculate Mean Reciprocal Rank (MRR) at rank 1.
    Parameters:
        match_positions: list of positions of the correct match for different iterations
    Returns:
        float, the average MRR at rank 1
    """
    # Count the number of experiments where the match was found at the first position
    count_first_position_matches = match_positions.count(0)

    # If no matches were found at the first position, return None or another appropriate value
    if count_first_position_matches == 0:
        return None 

    # Calculate MR1 by dividing the number of first position matches by the total number of experiments
    mr1 = count_first_position_matches / len(match_positions)

    return mr1
