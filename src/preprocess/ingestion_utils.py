from music21 import converter, note
import numpy as np


def get_score_onsets(midi_file_path):
    """
    takes in a midi file path and returns a list of numpy arrays, each containing the onset times of the notes in a part
    """
    return [np.array([element.offset for element in part.flat.notes]) for part in converter.parse(midi_file_path).parts]

def get_score_note_lengths(midi_file_path, part_number):
    """
    takes in a midi file path and returns two lists of numpy arrays. In the first, each containing the duration of the notes in a part. In the second, each stores whether the event is a note or rest. 
    """
    score = converter.parse(midi_file_path)
    
    
    
    note_length = [np.array([element.duration.quarterLength for element in part.flat.notesAndRests]) for part in score.parts][part_number-1]
    is_note = [np.array([isinstance(element, note.Note) for element in part.flat.notesAndRests]) for part in score.parts][part_number-1]
    starting_rest = 0
    if not is_note[0]:
        starting_rest = note_length[0]
        note_length = note_length[1:]
        is_note = is_note[1:]
    
    result = []
    i = 0
    while i < len(note_length)-1:
        if not is_note[i]:
            i += 1
            continue
        if not is_note[i+1]:
            result.append((note_length[i], note_length[i+1]))
            i += 1
        else:
            result.append((note_length[i], 0))
            i += 1
    result.append((note_length[-1], 0))
    print(result)
    return np.array(result)

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

# Example usage:
if __name__ == "__main__":
    # result = analyze_duration(3/4)  # Example: 1.5 quarter notes (a dotted quarter note)
    # print(result)  # Expected output: (1.0, 0.5, True, False)
    # print(get_onset_times("Dataset/05_Entertainer_tpt_tpt/Sco_05_Entertainer_tpt_tpt.mid"))
    print(get_performance_onsets("Dataset/06_Entertainer_sax_sax/Notes_1_sax_06_Entertainer.txt"))