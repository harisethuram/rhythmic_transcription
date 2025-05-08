# takes in tokenized barlineless-sequence and outputs a barlined sequence
import os
import torch
import argparse
from tqdm import tqdm
import json
from math import ceil, floor

from src.const_tokens import *
from src.utils import open_processed_data_dir, flatten_notes
from src.preprocess.tokenizer_utils import dur_to_notes
from src.note.Note import Note

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_barlines(input_sequence, token_to_id, id_to_token, beats_per_bar, offset, device=DEVICE):
    offset %= beats_per_bar
    if offset == 0:
        offset = beats_per_bar
    
    # add a barline, starting from the offset at every beats_per_bar
    input_sequence = [s for s in input_sequence if s not in CONST_TOKENS]
    note_lens = [id_to_token[s].get_len() for s in input_sequence]
    
    # create start_end list
    start_end = [(0, note_lens[0], False, id_to_token[input_sequence[0]])] # (onset, offset, to_be_tied_forward, token_id)
    
    for i in range(1, len(note_lens)):
        start_end.append([start_end[-1][1], start_end[-1][1] + note_lens[i], False, id_to_token[input_sequence[i]]])
    # print(start_end)
    m = offset
    
    i = 0
    results = []
    while i < len(start_end):
        if start_end[i][1] > m: # split it into two notes
            curr = start_end[i]
            start_end.pop(i)
            start_end.insert(i, (curr[0], m, True, Note(string=str(curr[3]))))
            start_end.insert(i + 1, (m, curr[1], False, Note(string=str(curr[3]))))
        
        results.append(start_end[i])
        if start_end[i][1] == m:
            results.append(BARLINE_TOKEN)
            m += beats_per_bar
            
        i += 1

    result_notes = []
    for i, result in enumerate(results):
        if result == BARLINE_TOKEN:
            result_notes.append(BARLINE_TOKEN)
        else:
            note_result = result[3]
            # note.tied_forward = result[2] if not note.is_rest else False
            notes = dur_to_notes(result[1] - result[0])
            
            if len(notes) == 1:
                notes[0].tied_forward = result[2]
            else:
                notes[0].tied_forward = True
                notes[1].tied_forward = result[2]
            
            
            for note in notes:
                # if note_result.is_rest:
                #     note.tied_forward = False
                note.is_rest = note_result.is_rest
                note.fermata = note_result.fermata
                note.staccato = note_result.staccato
                result_notes.append(note)
            # note.duration = duration[0]
            # note.dotted = duration[2]
            # note.triplet = duration[3]
           
    # make sure any rests are not tied forward
    for i in range(len(result_notes)):
        if isinstance(result_notes[i], Note) and result_notes[i].is_rest:
            result_notes[i].tied_forward = False
    

    return [token_to_id[note] if note in token_to_id.keys() else token_to_id[UNKNOWN_TOKEN] for note in result_notes ]
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add barlines to a sequence")
    parser.add_argument("--model_path", type=str, help="Path to language model")
    parser.add_argument("--input_path", type=str, help="Path to input json file")
    parser.add_argument("--data_dir", type=str, help="Path to processed data directory")
    parser.add_argument("--output_dir", type=str, help="Path to output directory")
    parser.add_argument("--beats_per_bar", type=int, default=4, help="Number of beats per bar")
    parser.add_argument("--offset", type=int, default=0, help="Offset for the barlines")
    args = parser.parse_args()    
    
    # # Load
    model = torch.load(args.model_path).to(DEVICE)
    os.makedirs(args.output_dir, exist_ok=True)
    
    token_to_id, id_to_token, _ = open_processed_data_dir("processed_data/all/barlines")
    
    with open(args.input_path, "r") as f:
        input_sequence = flatten_notes(json.load(f))
    
    input_sequence = [token_to_id[note] for note in input_sequence]
    
    candidate_beats_per_bar = [2, 2.5, 3, 3.5, 4, 5, 6, 7]
    highest_ll = -float("inf")
    best = None
    best_output_sequence = None
    
    for beats_per_bar in candidate_beats_per_bar:
        candidate_offsets = [i for i in range(0, beats_per_bar, 0.25)]
        for offset in candidate_offsets:
            output_sequence = torch.Tensor(add_barlines(input_sequence, token_to_id, id_to_token, beats_per_bar, offset)).to(DEVICE).long()
            output, _ = model(output_sequence)[:-1, ...]
            
            # compute the log likelihood
            ll = torch.log_softmax(output, dim=-1)
            output_sequence = output_sequence[1:]
            ll = torch.gather(ll, 1, output_sequence.unsqueeze(1).to(torch.int64)).squeeze().sum().item()
            
            if ll > highest_ll:
                highest_ll = ll
                best = (beats_per_bar, offset)
                best_output_sequence = output_sequence
                
    print(f"Best beats per bar: {best[0]}, offset: {best[1]}")
    print(f"Highest log likelihood: {highest_ll}")
    print(f"Best output sequence: {best_output_sequence}")
    
    best_metadata = {
        "beats_per_bar": best[0],
        "offset": best[1],
        "log_likelihood": highest_ll
    }
    # save to output directory
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "output.json"), "w") as f:
        json.dump(best_output_sequence.tolist(), f)
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(best_metadata, f)
        
    # just testing now
    # input_sequence = [Note(duration=4, dotted=False, triplet=False, fermata=False, staccato=False, tied_forward=False, is_rest=False), Note(duration=1, dotted=False, triplet=False, fermata=False, staccato=False, tied_forward=False, is_rest=False), Note(duration=2, dotted=False, triplet=False, fermata=False, staccato=False, tied_forward=False, is_rest=False), Note(duration=1, dotted=False, triplet=False, fermata=False, staccato=False, tied_forward=False, is_rest=False), Note(duration=4, dotted=True, triplet=False, fermata=False, staccato=False, tied_forward=False, is_rest=False)]
    # input_sequence = [Note(duration=6.75, dotted=False, triplet=False, fermata=False, staccato=False, tied_forward=False, is_rest=False)]
    # for i in input_sequence:
    #     print(i)