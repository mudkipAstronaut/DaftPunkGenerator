from music21 import *
import os
import numpy as np

#to understand and graph the initial data
from collections import Counter
import matplotlib.pyplot as plt

def read_midi(file):
    print("Loading Music File: ", file)

    notes = []
    notes_to_parse = None

    #parsing a midi file
    midi = converter.parse(file)

    #grouping based on different instruments
    s2 = instrument.partitionByInstrument(midi)

    #looping over all instruments
    for part in s2.parts:
        #select elements of only piano
        if 'Piano' in str(part):
            notes_to_parse = part.recurse()

            #finding whether a particular element is note or chord
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
    return np.array(notes)

if __name__ == "__main__":
    #load midi files

    #specify path
    path='smallData/'

    #read all the filenames
    files=[i for i in os.listdir(path) if i.endswith(".mid")]

    #reading each midi
    notes_array = np.array([read_midi(path+i) for i in files])

    notes_ = [element for note_ in notes_array for element in note_]

    #num of unique notes
    unique_notes = list(set(notes_))
    #print(len(unique_notes))


    #computing frequency of each note
    freq = dict(Counter(notes_))

    #consider only the frequencies
    no=[count for _, count in freq.items()]

    #set the figure size
    #plt.figure(figsize=(5,5))
    #plt.hist(no)
    #plt.show()

    #threshold
    thresh = 15
    frequent_notes = [note_ for note_, count in freq.items() if count >= thresh]
    #print(len(frequent_notes))

    #prep new music w only the top freqs
    new_music=[]
    for notes in notes_array:
        temp=[]
        for note_ in notes:
            if note_ in frequent_notes:
                temp.append(note_)
        new_music.append(temp)

    new_music = np.array(new_music)

    #preping the data
    no_timesteps = 32
    x = []
    y = []

    for note_ in new_music:
        for i in range(0, len(note_) - no_timesteps, 1):
            #preparing input/output sequence
            input_ = note_[i:i+no_timesteps]
            output_ =  note_[i+no_timesteps]

            x.append(input_)
            y.append(output_)
    
    x=np.array(x)
    y=np.array(y)
    
    unique_x = list(set(x.ravel()))
    x_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_x))

    x_seq=[]
    for i in x:
        temp=[]
        for j in i:
            #assign unique integer to every note
            temp.append(x_note_to_int[j])
        x_seq.append(temp)
    x_seq=np.array(x_seq)

    unique_y = list(set(y))
    y_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_y))
    y_seq=np.array([y_note_to_int[i] for i in y])

    #we will preserve 80% of data for training and 20% for evaluation
    
