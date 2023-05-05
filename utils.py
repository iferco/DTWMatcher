import os 
import pickle
import numpy as np
import pandas as pd


#Modify as needed:
PATH_TO_DF='musicnet_metadata.csv'
musicnet_df=pd.read_csv(PATH_TO_DF)



def random_schubert_composition_performance():
    """
    Get a random Schubert composition. For testing purposes with a smaller dataset.
    """
    path_to_schubert = 'data/Schubert/performance'

    #Inside of path_to_schubert there are composition folders with many wav files. Pick a random one and print file name
    random_schubert = np.random.choice(os.listdir(path_to_schubert))
    #pick random .wav file from this path
    perf_path = np.random.choice(os.listdir(os.path.join(path_to_schubert, random_schubert)))
    #get full path to this file
    performance_path = os.path.join(path_to_schubert, random_schubert, perf_path)
    #print('Performance file: ', performance_path)
    #Now we will get all possible midi files from the Schubert folder



    #print('Number of midi files to match our objective performance to: ', len(midi_files))
    #The performance we seek is:
    file_name=performance_path.split('/')[-1]
    name=get_composition(file_name.split('.')[0])
    #get from filename the movement of the composition
    movement=musicnet_df.loc[musicnet_df['id'] == int(file_name.split('.')[0])]['movement'].values[0]
    return (performance_path)


def random_composition_performance():
    """
    Get a random  composition from the df. Requires the csv of the dataset to be located in the same folder as this script.
    This works with MusicNet dataset, the needed file is called musicnet_metadata.csv
    """
    list=['Bach','Beethoven','Brahms', 'Cambini', 'Dvorak', 'Faure', 'Haydn','Mozart', 'Ravel', 'Schubert']
    #get random element from list
    compositor=np.random.choice(list)
    path_to_compositor = 'data/'+compositor+'/performance'

    #Inside of path_to_schubert there are composition folders with many wav files. Pick a random one and print file name
    random_schubert = np.random.choice(os.listdir(path_to_compositor))
    #pick random .wav file from this path
    perf_path = np.random.choice(os.listdir(os.path.join(path_to_compositor, random_schubert)))
    #get full path to this file
    performance_path = os.path.join(path_to_compositor, random_schubert, perf_path)
    #print('Performance file: ', performance_path)
    #Now we will get all possible midi files from the Schubert folder



    #print('Number of midi files to match our objective performance to: ', len(midi_files))
    #The performance we seek is:
    file_name=performance_path.split('/')[-1]
    name=get_composition(file_name.split('.')[0])
    #get from filename the movement of the composition
    movement=musicnet_df.loc[musicnet_df['id'] == int(file_name.split('.')[0])]['movement'].values[0]
    return (performance_path)


##################################
#           AUXILIARY            #


def get_path(id,midi_list):
    """
    Param: 
        id of the composition, an integer number.
        midi_list: list of all synthesized data paths.
    Returns:
        path of the composition with the given id.
    Get the path of a given ID.
    """
    #Go into data/synthesized data and look for the folder with the same name as the id

    for i in range(len(midi_list)):
        if midi_list[i].split('/')[-1].split('_')[0]==str(id):
            return midi_list[i]

def get_filtered_list(pruned_list ,midi_list):
    """
    Param: 
        pruned_list: list of pruned midis, each element of the list is a tuple id, cost.
        midi_list: full list of all synthesized data paths.
    Returns:
        filtered_list: list of paths to pruned midis.
    """
    filtered_list=[]
    for i in range(len(pruned_list)):
        filtered_list.append(get_path(pruned_list[i][0],midi_list))
    return filtered_list


#Return value of composition of row  with given id in the dataframe
def get_composition(id):
    """
    Param: id of the composition, an integer number.
    Get the composition from a given ID.
    """
    return musicnet_df.loc[musicnet_df['id'] == int(id)]['composition'].values[0]

#get composer
def get_composer_id(id):
    """
    Param: id of the composition, an integer number.
    Get the composer from a given ID.
    """
    return musicnet_df.loc[musicnet_df['id'] == int(id)]['composer'].values[0]

def get_composer_composition(composition):
    """
    Param: Composition.
    Get the compositor from a given composition.
    """
    return musicnet_df.loc[musicnet_df['composition'] == composition]['composer'].values[0]
def get_id(composition):
    return musicnet_df.loc[musicnet_df['composition'] == composition]['id'].values[0]

#From path to file, obtain both the composition and the movement
def get_full_name(path):
    """
    Param: 
        Full path of a file.
    Get the full name of the composition and the movement from a given path.
    Returns:
        A tuple (Composition, Movement).
    """
    file_name=path.split('/')[-1]
    id=file_name.split('.')[0]
    id=id.split('_')[0]
    
    return musicnet_df.loc[musicnet_df['id'] == int(id)]['composition'].values[0], musicnet_df.loc[musicnet_df['id'] == int(id)]['movement'].values[0]

def get_full_name_id(id):
    return musicnet_df[musicnet_df['id'] == int(id)]['composition'].values[0], musicnet_df.loc[musicnet_df['id'] == int(id)]['movement'].values[0]

def get_position(list,composition, movement):
    """
    Param: 
        list: Tuple list of compositions and movements.
        composition: Composition of the wanted element.
        movement: Movement of the wanted element.
    Returns: 
        position of the element in the list. Or -1 if not found.
    """
    for i in range(len(list)):
        if list[i][0]==composition and list[i][1]==movement:
            return i
    return -1

def save_pk_file(name, list):
    """
    Param: name of the file to save, list to save.
    Saves a list in a file with the given name.
    """
    with open(name+'.pk', 'wb') as f:
        pickle.dump(list, f, pickle.HIGHEST_PROTOCOL)
def save_txt_file(time, experiment):
    """
    Saves the time took for an experiment in a txt file.
    """
    with open("Timings.txt", "a") as f:
     #add text not delete previous
        f.write("\n Elapsed time with "+experiment+" : {:.2f} seconds".format(time))


 
def get_random_wav():
    """    
    Get a random wav file from the synthesized data.
    """

    path_to_wav='data/synthesized_data'
    wav_files = [f for f in os.listdir(path_to_wav) if f.endswith('.wav')]
    random_wav = np.random.choice(wav_files)
    return random_wav


def get_synthesized_data_list():
    """
    Get a list of all the full path to the synthesized midi files in the dataset.
    """
    path_to_midi='data/synthesized_data'
    midi_files = [os.path.join(path_to_midi, f) for f in os.listdir(path_to_midi) if f.endswith('.wav')]
    return midi_files