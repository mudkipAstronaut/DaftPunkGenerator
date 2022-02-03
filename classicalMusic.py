from music21 import *
import os
import numpy as np

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
    path='path/'

    #read all the filenames
    files=[i for i in os.listdir(path) if i.endswith(".mid")]

    #reading each midi
    notes_array = np.array([read_midi(path+i) for i in files])
