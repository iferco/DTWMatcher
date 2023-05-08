import numpy as np
import librosa
from scipy.signal import get_window
import os
from utils import *

class DTWMatcher:

    def __init__(self, database):
        self.database = database

    def extract_features(self, audio_frame, sr=44100, stride=512, n_fft=2048):
        chroma_features = librosa.feature.chroma_stft(y=audio_frame, sr=sr, tuning=0, norm=2,
                                                hop_length=stride, n_fft=n_fft)
        
        chroma_logch = librosa.power_to_db(chroma_features, ref=chroma_features.max())
        return chroma_logch

    

    def compute_dtw(self, audio_features, score_features):
        D, wp = librosa.sequence.dtw(X=audio_features, Y=score_features, metric='euclidean')
        return D,wp

    def average_distance(self,D, wp):
        path_length = len(wp)
        distance = sum([D[wp[i][0], wp[i][1]] for i in range(path_length)])
        avg_distance = distance / path_length
        return avg_distance
    
    def identify_score(self, audio_frames):
        results = []
        count=0
        for score_id, score_features in self.database.items():
            print(count)
            count+=1 # just to keep track of how long it takes

            dtw_matrix, wp = self.compute_dtw(audio_frames, score_features)
            cost=self.average_distance(dtw_matrix, wp)
            results.append((score_id, cost))

        return sorted(results, key=lambda x: x[1])


#Outside class
def create_database(MatcherInstance, audio_folder, duration=None):
    """
    Creates a dictionary to store the hashed CQTs of the database.
    Params:
        audio_folder: Path to the folder containing the audio files.
    """
    hashed_cqt_dict = {}

    for file_name in os.listdir(audio_folder):
        if file_name.endswith(".wav"):
            file_path = os.path.join(audio_folder, file_name)
            audio_frame, sr = librosa.load(file_path, sr=44100, duration=duration)

            if duration is None:
                features = MatcherInstance.extract_features(audio_frame, sr=44100)
            else:
                features = MatcherInstance.extract_features(audio_frame, sr=44100)
            
            id = file_name.split(".")[0]
            id = id.split("_")[0]
            print(id, " ", features.shape)

            hashed_cqt_dict[id] = features

    print("Feature dict created")
    return hashed_cqt_dict





audio_folder='./DiscoMIDI/data/synthesized_data'

audio_folder='../../synthesized_midis/'
file_paths=[]
input_file_path = random_composition_performance()
id = input_file_path.split(".")[0]
id = id.split("/")[-1]
print("Looking for ", input_file_path, "id: ", id)

# Load the audio data from the input file
input_audio, sr = librosa.load(input_file_path, sr=44100, duration=35)
My_DTW_instance = DTWMatcher(database={})
database = create_database(My_DTW_instance,audio_folder,duration=35)

My_DTW_instance = DTWMatcher(database=database)
input_features = My_DTW_instance.extract_features(input_audio, sr=44100)
my_list=My_DTW_instance.identify_score(input_features)
#print top 10
for i in range(10):
    print(my_list[i])
print(id)
print(my_list[0])
#

