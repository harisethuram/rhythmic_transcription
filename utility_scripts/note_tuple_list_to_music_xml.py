from typing import List
from music21 import stream, note, duration, articulations, expressions, tie, meter
import os
import json
from math import ceil
import argparse
# from .const_tokens import *

def convert_note_tuple_list_to_music_xml(note_tuple_list: List, output_dir, pitches=None):
    """
    Convert a list of note tuples to a MusicXML file for visualization
    If pitches is not passed, default to middle C for all notes
    """
    # if pitches is not None:
    #     assert len(note_tuple_list) == len(pitches)
    if pitches is None:
        pitches = ["C4"] * len(note_tuple_list)
        
    # convert this to a music21 stream
    s = stream.Stream()
    m = stream.Measure(number=1)
    pitch_pointer = 0
    
    
    total_length = 0
    
    for (
        qLength, 
        is_dotted, 
        is_triplet, 
        has_fermata, 
        is_staccato, 
        is_tied_forward, 
        is_rest) in note_tuple_list:

        # 1) Create Note or Rest
            
        if not is_rest:
            n = note.Note(pitches[pitch_pointer])
        else:
            n = note.Rest()
            pitch_pointer -= 1
            
        
        
        
        
        # 2) Set duration
        dur = duration.Duration(qLength)
        
        # If it's dotted
        if is_dotted:
            dur.dots = 1
        
        # If it's triplet
        # (There are a few ways to handle tuplets; 
        #  this is a simplistic example that treats triplets 
        #  as 3 notes in the time of 2.)
        if is_triplet:
            tuplet = duration.Tuplet(3, 2)
            # Alternatively: 
            # tuplet = duration.Tuplet(actual=3, normal=2)
            dur.appendTuplet(tuplet)
        
        n.duration = dur
        total_length += qLength * (3/2 if is_dotted else 1) * (2/3 if is_triplet else 1)
        # 3) Add articulations/expressions
        if has_fermata:
            ferm = expressions.Fermata()
            n.expressions.append(ferm)
        
        if is_staccato:
            stac = articulations.Staccato()
            n.articulations.append(stac)
        
        # 4) Tie handling
        if is_tied_forward:
            n.tie = tie.Tie('start')  # or 'continue', 'stop', etc.
            pitch_pointer -= 1
        
        # 5) Add to the stream
        m.append(n)
        pitch_pointer += 1
    # print(total_length)
    ts = meter.TimeSignature(f"{ceil(total_length)}/4")
    m.insert(0, ts)
    s.append(m)

    # 6) Write out to MusicXML
    os.makedirs(output_dir, exist_ok=True)
    
    s.write('musicxml', fp=os.path.join(output_dir, 'output.xml'))
        
        # s.write('musicxml', fp='output.xml')`
        
if __name__ == "__main__":
    # tmp = [(1, False, False, False, False, False, False), (1, False, False, False, False, False, False), (2, False, False, False, False, False, False), (1, False, False, False, False, False, False), (1, False, False, False, False, False, False), (2, False, False, False, False, False, False)]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input json file")
    parser.add_argument("--note_info_path", type=str, required=True, help="Path to the note info json file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    
    args = parser.parse_args()
    
    input_path = args.input_path
    with open(input_path, "r") as f:
        tmp = json.load(f)
    note_info_path = args.note_info_path
    with open(note_info_path, "r") as f:
        note_info = json.load(f)
        
    if tmp[0][0] == "<SOS>":
        tmp = tmp[1:]
    pitches = [note[-1] for note in note_info]
    # print(tmp[:5])
    output_dir = "test/"
    convert_note_tuple_list_to_music_xml(tmp, args.output_dir, pitches)
    