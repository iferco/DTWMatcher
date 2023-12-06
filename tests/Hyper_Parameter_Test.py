from utils import *
import librosa
import os
from DTWMatcher import *
from eval import *
import time
import pickle



audio_folder='./data/synthesized_data'
duration_l=[1024,2048]
frame_size=[1024,2048]
hop_size=[1024,2048]
name='results_hyperparameters_plain_cqt.txt'
pickle_name=name.split(".")[0]
mode='plain_cqt'

def test_base_dtw(duration,mode='chroma', hop_size=256, frame_size=512, iterations=100):
        length=[]
        position_list=[]
        chosen_id=[]
        for i in range(iterations):
            #print("Testing for duration of: ", duration)
            input_file_path = random_composition_performance()
            #exception with some compositions:
            
            id = input_file_path.split(".")[0]
            id = id.split("/")[-1]
            
            chosen_id.append(id)
            #print("Looking for ", input_file_path, "id: ", id)

            # Load the audio data from the input file
            input_audio, sr = librosa.load(input_file_path, sr=44100, duration=duration)
            if i == 0:
                # Create a DTWMatcher instance
                My_DTW_instance = DTWMatcher(database={})
                # This function returns a dictionary containing the chroma features of the database
                # The keys are the ids of the scores and the values are the chroma features

                database = create_database(My_DTW_instance,audio_folder,duration=duration, mode=mode, n_fft=frame_size, stride=hop_size)


            My_DTW_instance = DTWMatcher(database=database)
            #star timer
            st = time.process_time()
            # Extract the features of the input audio
            input_features = My_DTW_instance.extract_features(input_audio, sr=44100, mode=mode, n_fft=frame_size, stride=hop_size)
            # Identify the score with the lowest distance to the input audio
            my_list=My_DTW_instance.identify_score(input_features)


            #end timer
            et = time.process_time()

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

        return length, position_list, chosen_id
            

#Create txt file with results using with
with open(name, 'w') as f:
     #write Results: \n
     f.write("Results: \n") 
    


#test for different durations using hop size and frame size lists

for i in range(len(frame_size)): 
    length, position_list,chosen_ids=test_base_dtw(1,hop_size=hop_size[i],frame_size=frame_size[i],mode=mode)
    

    #Evaluation of 1 second results:
    print("Evaluation of 1 second results:" + str(hop_size[i]) + " " + str(frame_size[i]) + " : \n")

    print("Accuracy: ", get_accuracy(position_list), " %")
    print("NCDG: ", ndcg_at_k(position_list))

    print("Average time: ", sum([i[1] for i in length])/len(length))
    print("Number of top 10 matches: ", top_10_num_matches(position_list))
    print("Average of top 10 matches: ", mean_top_10_pos(position_list))
    #write results to txt file but do not overwrite
    with open(name, 'a') as f:
        f.write("Evaluation of 1 second results:"+"for "+str(hop_size[i])+ "and window size "+ str(frame_size[i]) +"\n")
        f.write("Accuracy: "+ str(get_accuracy(position_list))+ " % \n")
        f.write("Average time: "+ str(sum([i[1] for i in length])/len(length))+ "\n")
        f.write("Number of top 10 matches: "+ str(top_10_num_matches(position_list))+ "\n")
        f.write("Average of top 10 matches: "+ str(mean_top_10_pos(position_list))+ "\n")
        f.write("P@10: " + str(calculate_p_at_10(position_list)) + "\n")
        f.write("MR1: "+ str(calculate_mr1(position_list))+ "\n")
        f.write("MRR: "+ str(calculate_mrr(position_list))+ "\n")
        f.write("\n")

#save results to pickle file
with open(pickle_name+"_1.pickle", 'wb') as f:
    pickle.dump([length, position_list,chosen_ids], f)





for i in range(len(frame_size)):
    length, position_list,chosen_ids=test_base_dtw(5,mode=mode, hop_size=hop_size[i], frame_size=frame_size[i])

    #Evaluation of 5 second results:
    print("Evaluation of 5 second results:")

    print("Accuracy: ", get_accuracy(position_list), " %")
    print("NCDG: ", ndcg_at_k(position_list))

    print("Average time: ", sum([i[1] for i in length])/len(length))
    print("Number of top 10 matches: ", top_10_num_matches(position_list))
    print("Average of top 10 matches: ", mean_top_10_pos(position_list))
    #write results to txt file but do not overwrite
    with open(name, 'a') as f:
        f.write("Evaluation of 5 second results"+ str(hop_size[i]) + " " + str(frame_size[i]) + " : \n")
        f.write("Accuracy: "+ str(get_accuracy(position_list))+ " % \n")
        f.write("Average time: "+ str(sum([i[1] for i in length])/len(length))+ "\n")
        f.write("Number of top 10 matches: "+ str(top_10_num_matches(position_list))+ "\n")
        f.write("Average of top 10 matches: "+ str(mean_top_10_pos(position_list))+ "\n")
        f.write("P@10: " + str(calculate_p_at_10(position_list)) + "\n")
        f.write("MR1: "+ str(calculate_mr1(position_list))+ "\n")
        f.write("MRR: "+ str(calculate_mrr(position_list))+ "\n")
        f.write("\n")

#save results to pickle file
with open(pickle_name+"_5.pickle", 'wb') as f:
    pickle.dump([length, position_list,chosen_ids], f)

for i in range(len(frame_size)):
    length, position_list,chosen_ids=test_base_dtw(15,mode=mode, hop_size=hop_size[i], frame_size=frame_size[i])

    #Evaluation of 15 second results:
    print("Evaluation of 15 second results ", str(hop_size[i])," ", str(frame_size[i]) , " : \n")
    print("Number of top 10 matches: ", top_10_num_matches(position_list))
    print("Average of top 10 matches: ", mean_top_10_pos(position_list))


    print("Accuracy: ", get_accuracy(position_list), " %")
    print("NCDG: ", ndcg_at_k(position_list))

    print("Average time: ", sum([i[1] for i in length])/len(length))
    print("Number of top 10 matches: ", top_10_num_matches(position_list))
    print("Average of top 10 matches: ", mean_top_10_pos(position_list))
    #write results to txt file but do not overwrite
    with open(name, 'a') as f:
        f.write("Evaluation of 15 second results" + str(hop_size[i]) + " " + str(frame_size[i]) + " : \n")
        f.write("Accuracy: "+ str(get_accuracy(position_list))+ " % \n")
        f.write("Average time: "+ str(sum([i[1] for i in length])/len(length))+ "\n")
        f.write("Number of top 10 matches: "+ str(top_10_num_matches(position_list))+ "\n")
        f.write("Average of top 10 matches: "+ str(mean_top_10_pos(position_list))+ "\n")
        f.write("P@10: " + str(calculate_p_at_10(position_list)) + "\n")
        f.write("MR1: "+ str(calculate_mr1(position_list))+ "\n")
        f.write("MRR: "+ str(calculate_mrr(position_list))+ "\n")
        f.write("\n")
 
 #save results to pickle file
with open(pickle_name+"_15.pickle", 'wb') as f:
    pickle.dump([length, position_list,chosen_ids], f)

#Evaluation of 30 second results:
for i in range(len(frame_size)):
    length, position_list,chosen_ids=test_base_dtw(30,mode=mode, hop_size=hop_size[i], frame_size=frame_size[i])
    print("Evaluation of 30 second results:")

    print("Accuracy: ", get_accuracy(position_list), " %")
    print("NCDG: ", ndcg_at_k(position_list))

    print("Average time: ", sum([i[1] for i in length])/len(length))
    print("Number of top 10 matches: ", top_10_num_matches(position_list))
    print("Average of top 10 matches: ", mean_top_10_pos(position_list))
    #write results to txt file but do not overwrite
    with open(name, 'a') as f:
        f.write("Evaluation of 30 second results" + str(hop_size[i]) + " " + str(frame_size[i]) + " : \n")
        f.write("Accuracy: "+ str(get_accuracy(position_list))+ " % \n")
        f.write("Average time: "+ str(sum([i[1] for i in length])/len(length))+ "\n")
        f.write("Number of top 10 matches: "+ str(top_10_num_matches(position_list))+ "\n")
        f.write("Average of top 10 matches: "+ str(mean_top_10_pos(position_list))+ "\n")
        f.write("P@10: " + str(calculate_p_at_10(position_list)) + "\n")
        f.write("MR1: "+ str(calculate_mr1(position_list))+ "\n")
        f.write("MRR: "+ str(calculate_mrr(position_list))+ "\n")
        f.write("\n")
#save results to pickle file
with open(pickle_name+"_30.pickle", 'wb') as f:
    pickle.dump([length, position_list,chosen_ids], f)
