import numpy as np
import librosa
from scipy.signal import get_window
import os
from utils import *

class DTWMatcher:

    def __init__(self, database):
        self.database = database

    def extract_features(self, audio_frame, sr=44100, stride=512, n_fft=4096, mode='chroma'):
        """
        Extracts the chroma/CQT features from the audio frame.

        Params:
            audio_frame: Audio frame to extract the features from.
            sr: Sample rate of the audio frame.
            stride: Hoplength for the chroma.
            n_fft: FFT window size.
            Returns:
                features: Features extracted from the audio frame.
        """
        if mode == 'chroma':
            features = librosa.feature.chroma_stft(y=audio_frame, sr=sr, tuning=0, norm=2,
                                                    hop_length=stride, n_fft=n_fft)
            #features = librosa.power_to_db(features, ref=features.max())
            #Lin's work uses the amplitude in linear scale, so... let's return that
            #normalize the chroma features to have a mean of 0 and a variance of 1
            """
            features_mean = np.mean(features, axis=1, keepdims=True)
            features_std = np.std(features, axis=1, keepdims=True)
            features = (features - features_mean) / features_std
            """

            
        if mode == 'cqt':
            features = librosa.cqt(y=audio_frame, sr=sr, hop_length=stride, n_bins=252, bins_per_octave=36)


        return features

    

    def compute_dtw(self, audio_features, score_features):
        """
        Computes the DTW between the audio and score features using librosa's 
        implementation for DTW.

        Params:
            audio_features: Features extracted from the audio frame.
            score_features: Features extracted from the score frame.
            Returns:
                D: DTW matrix.
                wp: Warping path.
        """
        D, wp = librosa.sequence.dtw(X=audio_features, Y=score_features, metric='euclidean')
        return D,wp

    def average_distance(self,D, wp):
        """
        Computes the average distance between the audio and score features.

        Params:
            D: DTW matrix.
            wp: Warping path.
        Returns:
            avg_distance: Average distance between the audio and score features.
        """
        path_length = len(wp)
        distance = sum([D[wp[i][0], wp[i][1]] for i in range(path_length)])
        avg_distance = distance / path_length
        return avg_distance
    
    def identify_score(self, audio_frames):
        """
        Matches given query to all features in the database using DTW.
         
        Params:
            audio_frames: Audio frame to identify.
            Returns:
                results: List of tuples containing the score id and the average distance
                to the input audio frame ordered in ascendent value. The lowest distance,
                the most probable the match is.
        """
        results = []
        count=0
        for score_id, score_features in self.database.items():
            #print(count)
            count+=1 # just to keep track of how long it takes

            dtw_matrix, wp = self.compute_dtw(audio_frames, score_features)
            cost=self.average_distance(dtw_matrix, wp)
            results.append((score_id, cost))

        return sorted(results, key=lambda x: x[1])


#Outside class
def create_database(MatcherInstance, audio_folder, duration=None, n_fft=4096, stride=512, mode='chroma'):
    """
    Creates a dictionary to store the hashed chroma features of the database.

    Params:
        audio_folder: Path to the folder containing the audio files.
        duration: Duration of the audio files wanted to be loaded.
    Returns:
        features_dict: Dictionary containing the hashed chroma features of the database.

    """
    features_dict = {}

    for file_name in os.listdir(audio_folder):
        if file_name.endswith(".wav"):
            file_path = os.path.join(audio_folder, file_name)
            audio_frame, sr = librosa.load(file_path, sr=44100, duration=duration)

            if duration is None:
                features = MatcherInstance.extract_features(audio_frame, sr=44100, n_fft=n_fft, stride=stride, mode=mode)
            else:
                features = MatcherInstance.extract_features(audio_frame, sr=44100, n_fft=n_fft, stride=stride, mode=mode)
            
            id = file_name.split(".")[0]
            id = id.split("_")[0]
            #print(id, " ", features.shape)

            features_dict[id] = features

    print("Feature dict created")
    return features_dict




