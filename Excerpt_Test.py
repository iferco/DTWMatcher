from utils import *
import librosa
import os
from DTWMatcher import *
from eval import *
import time
import pickle



audio_folder='./data/synthesized_data'
duration_l=[1, 3, 6, 15, 35, 120]

mode='chroma'



#test for different durations
length, position_list=test_base_dtw(1)
#Evaluation of 1 second results:
print("Evaluation of 1 second results:")

print("Accuracy: ", get_accuracy(position_list), " %")
print("NCDG: ", ndcg_at_k(position_list))

print("Average time: ", sum([i[1] for i in length])/len(length))

 
length, position_list=test_base_dtw(5)
#Evaluation of 5 second results:
print("Evaluation of 5 second results:")

print("Accuracy: ", get_accuracy(position_list), " %")
print("NCDG: ", ndcg_at_k(position_list))

print("Average time: ", sum([i[1] for i in length])/len(length))

length, position_list=test_base_dtw(15)
#Evaluation of 15 second results:
print("Evaluation of 15 second results:")

print("Accuracy: ", get_accuracy(position_list), " %")
print("NCDG: ", ndcg_at_k(position_list))

print("Average time: ", sum([i[1] for i in length])/len(length))

#Evaluation of 35 second results:
length, position_list=test_base_dtw(35)
print("Evaluation of 35 second results:")

print("Accuracy: ", get_accuracy(position_list), " %")
print("NCDG: ", ndcg_at_k(position_list))

print("Average time: ", sum([i[1] for i in length])/len(length))