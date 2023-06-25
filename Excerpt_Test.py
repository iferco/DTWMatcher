from utils import *
import librosa
import os
from DTWMatcher import *
from eval import *
import time
import pickle



audio_folder='./data/synthesized_data'
duration_l=[1, 5, 15, 30]

mode='cqt'

def test_base_dtw(duration,mode='chroma', iterations=100):
        length=[]
        position_list=[]
        for i in range(iterations):
            #print("Testing for duration of: ", duration)
            input_file_path = random_composition_performance()
            #exception with some compositions:

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
            my_list=My_DTW_instance.identify_score(input_features, mode=mode)


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
            



#test for different durations
length, position_list=test_base_dtw(1,mode=mode)
save_pk_file("position_list_1s",position_list)
save_pk_file("length_1s",length)

#Evaluation of 1 second results:
print("Evaluation of 1 second results:")

print("Accuracy: ", get_accuracy(position_list), " %")

print("NCDG: ", ndcg_at_k(position_list))

print("Average time: ", sum([i[1] for i in length])/len(length))
print("Number of top 10 matches: ", top_10_num_matches(position_list))
print("Average of top 10 matches: ", mean_top_10_pos(position_list))
print("P@10: " + str(calculate_p_at_10(position_list)) + "\n")
print("MR1: "+ str(calculate_mr1(position_list))+ "\n")


length, position_list=test_base_dtw(5,mode=mode)
save_pk_file("position_list_5s", position_list)
save_pk_file("length_5s",length)

#Evaluation of 5 second results:
print("Evaluation of 5 second results:")
print("Number of top 10 matches: ", top_10_num_matches(position_list))
print("Average of top 10 matches: ", mean_top_10_pos(position_list))
print("Accuracy: ", get_accuracy(position_list), " %")
print("NCDG: ", ndcg_at_k(position_list))

print("Average time: ", sum([i[1] for i in length])/len(length))
print("P@10: " + str(calculate_p_at_10(position_list)) + "\n")
print("MR1: "+ str(calculate_mr1(position_list))+ "\n")

length, position_list=test_base_dtw(15,mode=mode)
save_pk_file("position_list_15s",position_list)
save_pk_file("length_15s",length)

#Evaluation of 15 second results:
print("Evaluation of 15 second results:")
print("Number of top 10 matches: ", top_10_num_matches(position_list))
print("Average of top 10 matches: ", mean_top_10_pos(position_list))


print("Accuracy: ", get_accuracy(position_list), " %")
print("NCDG: ", ndcg_at_k(position_list))

print("Average time: ", sum([i[1] for i in length])/len(length))
print("Number of top 10 matches: ", top_10_num_matches(position_list))
print("Average of top 10 matches: ", mean_top_10_pos(position_list))
 
#Evaluation of 30 second results:
length, position_list=test_base_dtw(30,mode=mode)
save_pk_file("position_list_30s",position_list)
save_pk_file("length_30s",length)
print("Evaluation of 30 second results:")

print("Accuracy: ", get_accuracy(position_list), " %")
print("NCDG: ", ndcg_at_k(position_list))
print("P@10: " + str(calculate_p_at_10(position_list)) + "\n")
print("MR1: "+ str(calculate_mr1(position_list))+ "\n")

print("Average time: ", sum([i[1] for i in length])/len(length))
print("Number of top 10 matches: ", top_10_num_matches(position_list))
print("Average of top 10 matches: ", mean_top_10_pos(position_list))
 

 