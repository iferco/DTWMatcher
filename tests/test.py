from utils import *
import librosa
import os
from DTWMatcher import *
from eval import *
import time

audio_folder='./data/synthesized_data'
duration=35
mode='chroma'

file_paths=[]
input_file_path = random_composition_performance()
id = input_file_path.split(".")[0]
id = id.split("/")[-1]
print("Looking for ", input_file_path, "id: ", id)

# Load the audio data from the input file
input_audio, sr = librosa.load(input_file_path, sr=44100, duration=duration)
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
print('Execution time:', elapsed_time, 'seconds')

#print top 10
for i in range(10):
    print(my_list[i])
print("We want id: ", id)
print("Our top match is: ", my_list[0])
print("Match in position: ", get_position_id(my_list, id))




