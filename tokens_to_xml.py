# convert list of tokens to xml
import json
import os
import music21
import argparse
from music21 import stream, note, duration, articulations, expressions, bar, tie
from fractions import Fraction

from src.utils import open_processed_data_dir
from src.const_tokens import *
from src.note.Note import Note

def tokens_to_stream(notes, pitches):
    s = stream.Stream()
    measure = stream.Measure()
    l = []
    for i, (tok, pitch) in enumerate(zip(notes, pitches)):
        # print(tok, pitch)
        if isinstance(tok, str):
            if tok.upper() == "<BARLINE>":
                s.append(measure)
                l.append(measure)
                measure = stream.Measure()
        else:
            curr_obj = note.Rest() if tok.is_rest else note.Note(pitch)
            
        
            quarter_length = tok.duration * (Fraction(2, 3) if tok.triplet else 1)
            curr_obj.duration = duration.Duration(quarter_length)
            if tok.dotted:
                curr_obj.duration.dots = 1
            if tok.staccato:
                curr_obj.articulations.append(articulations.Staccato())
            if tok.fermata:
                curr_obj.expressions.append(expressions.Fermata())
            if i > 0 and isinstance(notes[i-1], Note) and notes[i-1].tied_forward:
                curr_obj.tie = tie.Tie('continue' if tok.tied_forward else 'stop')
            elif tok.tied_forward:
                curr_obj.tie = tie.Tie('start')
            
            # print("curr_obj", curr_obj, curr_obj.duration)

            measure.append(curr_obj)
        # input()
    # print("num", len(l))
    s.append(measure)
    return s

def save_stream(s, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    s.write("xml", fp=output_path)
    print(f"Saved XML to {output_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSON file, output of add_barlines.py")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output XML file")
    args = parser.parse_args()
    
    # Load the input JSON file
    with open(args.input_path, "r") as f:
        data = json.load(f)
        
    notes = [Note(string=d) if d not in CONST_TOKENS else d for d in data["sequence"]]
    pitches = data["pitches"]
    
    output = tokens_to_stream(notes, pitches)
    save_stream(output, args.output_path)
    print(f"Saved XML to {args.output_path}")