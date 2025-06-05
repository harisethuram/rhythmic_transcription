from music21 import converter, note
import numpy as np
from librosa import hz_to_note
from fractions import Fraction


def get_note_info_from_xml(xml_file_path: str, part_id: int):
    """
    returns list of tuples of (onset length, note portion, rest portion, pitch)
    """
    score = converter.parse(xml_file_path)
    part = score.parts[part_id - 1]
    note_lengths = [Fraction(element.duration.quarterLength) for element in part.flat.notesAndRests]
    is_note = [isinstance(element, note.Note) for element in part.flat.notesAndRests]
    pitches = [element.name if isinstance(element, note.Note) else None for element in part.flat.notesAndRests]
    
    # rests are technically 'tied' to the next rest
    is_tied_forward = []
    for i, element in enumerate(part.flat.notesAndRests):
        if isinstance(element, note.Note):
            is_tied_forward.append(element.tie is not None and element.tie.type in ("start", "continue"))
            
        # if it's a rest, if it isn't the last element, check if the next element is a rest. if so, it's tied forward
        else:
            if i < len(part.flat.notesAndRests) - 1:
                is_tied_forward.append(isinstance(part.flat.notesAndRests[i + 1], note.Rest))
            else:
                is_tied_forward.append(False)
                
    
    # now remove leading rests
    while not is_note[0]:
        note_lengths = note_lengths[1:]
        is_note = is_note[1:]
        pitches = pitches[1:]
        is_tied_forward = is_tied_forward[1:]
        
    # now, merge tied forward notes
    i = 0
    while i < len(note_lengths):
        # if it's a note and it's tied forward, merge it with the next note
        if is_tied_forward[i]:
            note_lengths[i + 1] += note_lengths[i]
            note_lengths.pop(i)
            pitches.pop(i)
            is_note.pop(i)
            is_tied_forward.pop(i)
        else:
            i += 1
                
    
    # now convert to onset tuples
    onset_lengths = []
    onset_pitches = []
    onset_note_portions = []
    onset_rest_portions = []
    curr_onset_length = 0
    curr_onset_pitch = None
    curr_note_length = 0
    curr_rest_length = 0
    for i in range(len(note_lengths)):
        curr_onset_length += note_lengths[i]
        if is_note[i]:
            curr_note_length += note_lengths[i]
            curr_onset_pitch = pitches[i]
        else:
            curr_rest_length += note_lengths[i]
                    
        if i+1 < len(note_lengths) and is_note[i + 1]:
            assert curr_note_length + curr_rest_length == curr_onset_length
            onset_lengths.append(curr_onset_length)
            onset_pitches.append(curr_onset_pitch)
            try:
                onset_note_portions.append(Fraction(curr_note_length, curr_onset_length))
            except:
                print(curr_note_length, curr_onset_length, curr_onset_pitch)
            onset_rest_portions.append(Fraction(curr_rest_length, curr_onset_length))
            curr_onset_length = 0
            curr_onset_pitch = None
            curr_note_length = 0
            curr_rest_length = 0

    onset_lengths.append(curr_onset_length)
    onset_pitches.append(curr_onset_pitch)
    onset_note_portions.append(curr_note_length / curr_onset_length)
    onset_rest_portions.append(curr_rest_length / curr_onset_length)
    
    result = []
    for i in range(len(onset_lengths)):
        result.append([float(onset_lengths[i]), float(onset_note_portions[i]), float(onset_rest_portions[i]), onset_pitches[i]])
    return result            

    
    

def midi_to_onsets(midi_file_path: str):
    """
    takes in a midi file path and returns a list of numpy arrays, each containing the onset times of the notes in a part
    """
    return [np.array([element.offset for element in part.flat.notes]) for part in converter.parse(midi_file_path).parts]

def get_score_note_lengths(midi_file_path: str, part_number: int, want_last=False):
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