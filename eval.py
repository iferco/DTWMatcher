
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
            ndcgs.append(dcg / idcg)
        else:  # pos >= k
            ndcgs.append(0)
    return np.mean(ndcgs)



def test_base_dtw(duration,mode='chroma', iterations=100):
    """
    This function tests the baseline DTW matching algorithm for different durations of the input audio.
    Params:
        duration: seconds of the audio to be tested
        mode: chroma or cqt for baseline DTW
        iterations: number of iterations to perform the matching experiment.
    Returns:
        length: list of tuples with the duration of the audio and the time it took to match it
        position_list: list of positions of the correct match for different iterations
    """
    length=[]
    position_list=[]
    for i in range(iterations):
        #print("Testing for duration of: ", duration)
        input_file_path = random_composition_performance()
        id = input_file_path.split(".")[0]
        id = id.split("/")[-1]
        #print("Looking for ", input_file_path, "id: ", id)
        # Load the audio data from the input file
        input_audio, sr = librosa.load(input_file_path, sr=44100, duration=duration)
        if i == 0:
            # Create a DTWMatcher instance
            My_DTW_instance = DTWMatcher(database={})
            # This function returns a dictionary containing the chroma features of the database
            # The keys are the ids of the scores and the values are the chroma features
            database = create_database(My_DTW_instance,audio_folder,duration=duration, mode=mode)
            My_DTW_instance = DTWMatcher(database=database)
            #star timer
            st = time.time()
            # Extract the features of the input audio
            input_features = My_DTW_instance.extract_features(input_audio, sr=44100, mode=mode)
            # Identify the score with the lowest distance to the input audio
            my_list=My_DTW_instance.identify_score(input_features)


            #end timer
            et = time.time()

            # get the execution time
            elapsed_time = et - st
            length.append((duration, elapsed_time))
            #print('Execution time:', elapsed_time, 'seconds')
            """
            #print top 10
            for i in range(10):
                print(my_list[i])
            print("We want id: ", id)
            print("Our top match is: ", my_list[0])
            """
            position=get_position_id(my_list, id)
            #print("Match found in position: ", position )
            position_list.append(position)

        return length, position_list
            
