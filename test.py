from utils import *
import librosa
import os
from DTWMatcher import *


audio_folder='./data/synthesized_data'

file_paths=[]
input_file_path = random_composition_performance()
id = input_file_path.split(".")[0]
id = id.split("/")[-1]
print("Looking for ", input_file_path, "id: ", id)

# Load the audio data from the input file
input_audio, sr = librosa.load(input_file_path, sr=44100, duration=35)
# Create a DTWMatcher instance
My_DTW_instance = DTWMatcher(database={})
# This function returns a dictionary containing the chroma features of the database
# The keys are the ids of the scores and the values are the chroma features
database = create_database(My_DTW_instance,audio_folder,duration=35)


My_DTW_instance = DTWMatcher(database=database)
# Extract the features of the input audio
input_features = My_DTW_instance.extract_features(input_audio, sr=44100)
# Identify the score with the lowest distance to the input audio
my_list=My_DTW_instance.identify_score(input_features)

#print top 10
for i in range(10):
    print(my_list[i])
print(id)
print(my_list[0])


