from music21 import converter, note
import numpy as np


def get_score_onsets(midi_file_path):
    """
    takes in a midi file path and returns a list of numpy arrays, each containing the onset times of the notes in a part
    """
    return [np.array([element.offset for element in part.flat.notes]) for part in converter.parse(midi_file_path).parts]
 
def get_performance_onsets(input_path):
    """
    takes in a text file path and returns a numpy array of onset times
    """   
    with open(input_path, 'r') as f:
        return np.array([float(line.split("\t")[0]) for line in f.readlines()])

# Example usage:
if __name__ == "__main__":
    # result = analyze_duration(3/4)  # Example: 1.5 quarter notes (a dotted quarter note)
    # print(result)  # Expected output: (1.0, 0.5, True, False)
    # print(get_onset_times("Dataset/05_Entertainer_tpt_tpt/Sco_05_Entertainer_tpt_tpt.mid"))
    print(get_performance_onsets("Dataset/06_Entertainer_sax_sax/Notes_1_sax_06_Entertainer.txt"))