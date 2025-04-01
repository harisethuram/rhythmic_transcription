from music21 import converter, note
import numpy as np
from librosa import hz_to_note


def midi_to_onsets(midi_file_path):
    """
    takes in a midi file path and returns a list of numpy arrays, each containing the onset times of the notes in a part
    """
    return [np.array([element.offset for element in part.flat.notes]) for part in converter.parse(midi_file_path).parts]

def get_score_note_lengths(midi_file_path, part_number, want_last=False):
    """
    takes in a midi file path and returns a tuple consisitng of a numpy array of shape (n, 2) of note lengths and rest lengths, and a float of the starting rest if there is one
    """
    score = converter.parse(midi_file_path)
    
    note_lengths = np.array([element.duration.quarterLength for element in score.parts[part_number-1].flat.notesAndRests])
    is_note = np.array([isinstance(element, note.Note) for element in score.parts[part_number-1].flat.notesAndRests])
    print([(note_length, is_note_i) for note_length, is_note_i in zip(note_lengths, is_note)])
    starting_rest = 0
    
    if not is_note[0]:
        starting_rest = note_lengths[0]
        note_lengths = note_lengths[1:]
        is_note = is_note[1:]
    
    result = []
    i = 0
    while i < len(note_lengths)-1:
        if not is_note[i]:
            i += 1
            continue
        if not is_note[i+1]:
            result.append((note_lengths[i], note_lengths[i+1]))
            i += 1
        else:
            result.append((note_lengths[i], 0))
            i += 1
    if want_last:
        result.append((note_lengths[-1], 0))
        
    return np.array(result), starting_rest

def get_performance_onsets(input_path):
    """
    takes in a text file path and returns a numpy array of onset times
    """   
    with open(input_path, 'r') as f:
        return np.array([float(line.split("\t")[0]) for line in f.readlines()])

def get_performance_note_lengths(input_path):
    """
    takes in a text file path and returns a numpy array of tuples of (note length, rest length)
    """
    
    onset_lengths = np.diff(get_performance_onsets(input_path))
    
    with open(input_path, 'r') as f:
        note_lengths = np.array([float(line.strip().split("\t")[-1]) for line in f.readlines()])
        rest_lengths = onset_lengths - note_lengths[:-1]
        tuples_list = [(note_lengths[i], rest_lengths[i]) for i in range(len(note_lengths[:-1]))]
        tuples_list.append((note_lengths[-1], 0))
    
    
    result = np.array(tuples_list)
    assert np.allclose(np.sum(result, axis=1)[:-1], onset_lengths)
    return result

def get_performance_pitches(input_path):
    """
    takes in a text file path and returns a numpy array of note names
    """
    with open(input_path, 'r') as f:
        pitches = np.array([float(line.split("\t")[2]) for line in f.readlines()])
        return freq_to_note(pitches)


def freq_to_note(frequencies):
    """
    takes in a list of frequencies and returns a list of note names
    """
    note_names = []
    for freq in frequencies:
        note_names.append(hz_to_note(freq))
    note_names = [note_name.replace("♯", "#") for note_name in note_names]
    note_names = [note_name.replace("♭", "b") for note_name in note_names]
    return note_names



# Example usage:
if __name__ == "__main__":
    # result = analyze_duration(3/4)  # Example: 1.5 quarter notes (a dotted quarter note)
    # print(result)  # Expected output: (1.0, 0.5, True, False)
    get_score_note_lengths("data/URMP/val/01_Jupiter_vn_vc/Sco_01_Jupiter_vn_vc.mid", 2, True)
    # print(get_performance_onsets("Dataset/06_Entertainer_sax_sax/Notes_1_sax_06_Entertainer.txt"))