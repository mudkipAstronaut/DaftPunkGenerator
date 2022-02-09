import pickle
import glob
import numpy as np
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def train_network():
    notes = get_notes()

    #get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)
    model = create_network(network_input, network_output)

    train(model, network_input, network_output)

def get_notes():
    """extracts all the notes and chords of the midi files
    in the songs folder and creates a file w the notes in
    string format"""

    notes=[]

    for file in glob.glob("techSongs/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None
        try: #file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            print(str(s2.parts[0])
            notes_to_parse=s2.parts[0].recurse()
        except: #file hsa notes in a flat structure
            notes_to_parse = midi.flat.notes
        
        for element in notes_to_parse:
            if(isinstance(element, note.Note)):
                notes.append(str(element.pitch))
            elif(isinstance(element, chord.Chord)):
                notes.append('.'.join(str(n) for n in element.normalOrder))
        
    with open("data/notes", 'wb') as f:
        pickle.dump(notes, f)
    
    return notes

def prepare_sequences(notes, n_vocab):
    """Prepare sequences to be the inputs to the LSTM"""
    #sequence length should be changed after experimenting w diff numbers
    sequence_length = 30

    #get all pitch names
    pitchnames = sorted(set(item for item in notes))

    #create a dict to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input=[]
    network_output=[]

    #create input sequences and the corresponding outputs
    for i in range(0, len(notes)-sequence_length, 1):
        sequence_in = notes[i:i+sequence_length]
        sequence_out = notes[i+sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    #reshape input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length,1))
    #normalize input
    network_input = network_input/float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

def create_network(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, network_input, network_output):
    """train the neural net"""
    filepath = "data/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    #experiment w diff epoch size and batch sizes
    model.fit(
        network_input, network_output,
        epochs = 30
        batch_size=64,
        callbacks=callbacks_list
    )

if __name__=="__main__":
    train_network()