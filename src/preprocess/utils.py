import mido
from music21 import converter, note
import numpy as np

def analyze_duration(dur):
        """
        Analyzes the given float duration and returns a tuple with:
        - base note value (e.g., 1.0 for a quarter note)
        - dot value (e.g., 0.5 for a dotted note)
        - whether the note is dotted (True/False)
        - whether the note is a triplet (True/False)
        
        Parameters:
        float_duration (float): The duration in terms of quarter notes.
        
        Returns:
        tuple: (base_value, dot_value, is_dotted, is_triplet)
        """
        base_value = dur.quarterLength
        float_duration = dur.quarterLength
        dot_value = 0
        is_dotted = False
        is_triplet = False
        
        # Define common base note values (in quarter note lengths)
        note_values = [2**i for i in range(-3, 3)] # 4.0, 2.0, 1.0, 0.5, 0.25, 0.125]  # whole, half, quarter, eighth, sixteenth, thirty-second
        
        # Check for triplets first
        triplet_ratio = 2/3
        for value in note_values:
            if abs(float_duration - value * triplet_ratio) < 0.001:
                base_value = value
                is_triplet = True
                is_dotted = False
                dot_value = 0
                return base_value, dot_value, is_dotted, is_triplet
        
        # Check if the duration is a dotted note
        for value in note_values:
            if float_duration == value * 1.5:  # Dotted note
                base_value = value
                dot_value = value / 2
                is_dotted = True
                break
            elif float_duration == value:  # Non-dotted note
                base_value = value
                dot_value = 0
                is_dotted = False
                break

        return base_value, dot_value, is_dotted, is_triplet


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
    print(get_performance_onset_times("Dataset/06_Entertainer_sax_sax/Notes_1_sax_06_Entertainer.txt"))