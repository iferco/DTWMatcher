from utils import *
from pruning.prune import *
from matching.matching import *

#Testing some functions to get the needed data
print(random_schubert_composition_performance())
midi_files=get_synthesized_data_list()
print(midi_files[3])


#Testing the whole matching process!

print("\n PRUNING STAGE \n")
#The file we will look for will be:
perf_path=random_composition_performance()
full_perf_name=get_full_name(perf_path)
print(full_perf_name)
#First we need to create the dictionary of hashed cqt's
synthesized_data_path='data/synthesized_data'
"""
Since I already created the dictionary, I will just load it
hashed_cqt_dict=create_hashed_cqt_dict(synthesized_data_path)

#Now we need to process the cqt of the performance we want to find
"""
#load hashed_cqt_dict
hashed_cqt_dict = pickle.load(open("hashed_cqt_dict.pk", "rb"))
hashed_cqt = process_audio_file(perf_path)

#Let's start counting the full matching process
pruning_success=False
start = time.time()


top_matches = find_top_matches(hashed_cqt, hashed_cqt_dict)
        
    #print("Top 50 matches to ", get_full_name(input_audio_path), ":")
for match in top_matches:
    if get_full_name_id(match[0])==get_full_name(perf_path):
            print("Match found!")
            pruning_success=True
            break

if pruning_success:
    # start timer 
    

    print("Pruning succesful!")

    print("\nMATCHING STAGE\n")
    #Let's get our filtered list containing the elements obtained by pruning
    filtered_list=get_filtered_list(top_matches, midi_files)

    results=matching(filtered_list, perf_path, 15)
    results_names=[x[0] for x in results]
    print(results_names[:5])
    position=get_position(results_names, full_perf_name[0], full_perf_name[1])
    print(position)
    if position!=-1:
        print("Matching succesful! :)")
        #get list with only the names
        
        print("The performance was found in position ",position)
    else:
        print("Matching failed :(")
    #stop timer 
    end = time.time()
    # total time taken
    print(f"Runtime of the program is {end - start}")
else:
    print("Pruning failed :(")
    print("\nMATCHING STAGE\n")

    #Sadly we have to do the matching with the whole list
    results=matching(midi_files, perf_path, 35) 
    results_names=[x[0] for x in results]
    print(results_names[:5])
    position=get_position(results_names, full_perf_name[0], full_perf_name[1])
    print(position)
    if position!=-1:
        print("Matching succesful! :)")
        #get list with only the names
        
        print("The performance was found in position ", position)
    else:
        print("Matching failed :(")
    #stop timer
    end = time.time()
    # total time taken
    print(f"Runtime of the program is {end - start}")