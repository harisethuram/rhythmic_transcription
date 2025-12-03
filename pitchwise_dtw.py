# get alignment between pitch_1 and pitch_2 by DTW using pitch as loss. Can be many-many

import numpy as np
from typing import List
from math import inf
from music21 import pitch as m21pitch
import argparse

def get_dtw_alignment_pitchwise(pitch_1: List[str], pitch_2: List[str]) -> List[List[int]]:
    """
    Get DTW alignment between two pitch sequences using pitch as loss
    pitch_1: list of pitches (str) for sequence 1
    pitch_2: list of pitches (str) for sequence 2
    returns: alignment as list of [i, j] pairs where i is index in pitch_1 and j is index in pitch_2
    """
    
    pitch_1 = [m21pitch.Pitch(p) for p in pitch_1 if p is not None]
    pitch_2 = [m21pitch.Pitch(p) for p in pitch_2 if p is not None]
    
    def cost(p1, p2):
        # binary cost: 0 if same pitch, 1 otherwise
        return 0 if p1 == p2 else 1
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_midi_path", type=str, required=True, help="Path to ground truth MIDI file")
    parser.add_argument("--performance_midi_path", type=str, required=True, help="Path to performance MIDI file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output alignment file")
    args = parser.parse_args()
    
    
    
    
    
    