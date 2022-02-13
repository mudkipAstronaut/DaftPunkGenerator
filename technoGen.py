import pickle
import glob
import numpy as np
from music21 import converter, instrument, note, chord, stream
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint

import sys

def train_network():
    notes = get_notes()

    print("got notes")
    #get amount of pitch names
    n_vocab = len(set(notes))

    print("about to prepare sequences")
    network_input, network_output = prepare_sequences(notes, n_vocab)
    print("about to make network")
    model = create_network(network_input, n_vocab)

    print("about to train")
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

    network_output = utils.to_categorical(network_output)

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
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    #experiment w diff epoch size and batch sizes
    model.fit(
        network_input, network_output,
        epochs = 30,
        batch_size=64,
        callbacks=callbacks_list
    )

def prepare_sequences_pre(notes, pitchnames, n_vocab):
    #map back from ints to notes
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input=[]
    output=[]
    for i in range(0,len(notes)-sequence_length,1):
        sequence_in=notes[i:i+sequence_length]
        sequence_out=notes[i+sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])
    n_patterns = len(network_input)

    #reshape input into a format compatible w LSTM
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length,1))
    normalized_input = normalized_input/float(n_vocab)

    return (network_input, normalized_input)

def generate_notes(model, network_input, pitchnames, n_vocab):
    """Generate notes form the neural net based on a
    sequence of notes"""
    #pick a random sequence from the input as a starting point for the prediction
    start = np.random.randint(0,len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output=[]

    #generate 500 notes
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input/float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction) #numpy array of predictions
        result = int_to_note[index] #indexing the note w the highest probability
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    return prediction_output

def create_midi(prediction_output):
    offset = 0
    output_notes = []

    #create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        #pattern is chord
        if('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes=[]
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.SnareDrum()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else: #pattern is note
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.SnareDrum()
            output_notes.append(new_note)

        #increase offset each iteration so the notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output_song.mid')

def generate(weight_path):
    """gens the midi file"""
    notes = None
    filepath = 'data/notes'
    with open(filepath, 'rb') as f:
        notes = pickle.load(f)


    #get pitchnames
    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(set(notes))

    network_input, normalized_input = prepare_sequences_pre(notes, pitchnames, n_vocab)
    model = create_network(normalized_input, n_vocab)
    model.load_weights(weight_path)
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)

if __name__=="__main__":
    try:
        if(sys.argv[1] == "--train"):
            train_network()
        elif(sys.argv[1] == "--gen"):
            if(sys.argv[2]):
                generate(sys.argv[2])
            else:
                print("Music generation requires file path")
    except Exception as e:
        print(e)
        print("Options are --train or --gen filepath")