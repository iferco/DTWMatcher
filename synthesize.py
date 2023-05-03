from midi2audio import FluidSynth
"#First, we create the composition folders:\n",
import os
midi_list=[]
for root, dirs, files in os.walk("."):
    for dir in dirs:
        #print(dir)
        #Get from df all compositions who has dir as composer
        #get all files that end in .mid inside of dir
        #and print it
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(".mid"):
                    #print(os.path.join(root, file))
                    midi_list.append(os.path.join(root, file))


fs = FluidSynth()   

def synthesize(midi_path):
    midi_name=midi_path.split('/')[-1]
    audio_name=midi_name.split('.')[0]+'.wav'
    if 'WTK' in audio_name:
        midi_name=midi_path.split('/')[-1]
        audio_name=midi_name.split('.')[0]+'.wav'
        test=midi_name
        # Split the string by backslashes (\) to separate the folder names and the file name
        segments = test.split("\\")

        # Get the last segment, which contains the file name
        file_name = segments[-1]

        # Remove the ".mid" extension from the file name
        file_name_without_extension = file_name.replace(".mid", "")
        audio_name=file_name_without_extension+'.wav'

    fs.midi_to_audio(midi_path,audio_name)
    # Move the generated WAV file to the wav_songs folder
    os.rename(audio_name, 'wav_songs\\'+audio_name.split('\\')[-1])


"""
for midi in midi_list:
    try:
       synthesize(midi)


    #print exception
    except Exception as e:
        print(e, "in ", midi)
"""
"#First, we create the composition folders:\n",
import os
midi_list=[]
for root, dirs, files in os.walk("."):
    for dir in dirs:
        #print(dir)
        #Get from df all compositions who has dir as composer
        #get all files that end in .mid inside of dir
        #and print it
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(".mid") and ('fugue'  in file or 'prelude' in file):
                    #print(os.path.join(root, file))
                    midi_list.append(os.path.join(root, file))


for midi in midi_list:
    try:
       synthesize(midi)


    #print exception
    except Exception as e:
        print(e, "in ", midi)

