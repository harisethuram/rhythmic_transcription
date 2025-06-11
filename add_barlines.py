# takes in tokenized barlineless-sequence and outputs a barlined sequence
import os
import torch
import argparse
from tqdm import tqdm
import json
from math import ceil, floor

from src.const_tokens import *
from src.utils import open_processed_data_dir, flatten_notes, serialize_json
from src.preprocess.tokenizer_utils import dur_to_notes
from src.note.Note import Note
# from src.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from contextlib import contextmanager
import warnings

@contextmanager
def no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def add_barlines(input_sequence, pitches, token_to_id, id_to_token, beats_per_bar, offset, device=DEVICE):
    offset %= beats_per_bar
    if offset == 0:
        offset = beats_per_bar
    pitches = [pitch for pitch in pitches]
    # add a barline, starting from the offset at every beats_per_bar
    for s, p in zip(input_sequence, pitches):
        if id_to_token[s] in CONST_TOKENS:
            print("Error: token is in CONST_TOKENS", id_to_token[s], p)
    input_sequence = [s for s in input_sequence if id_to_token[s] not in CONST_TOKENS]
    note_lens = [id_to_token[s].get_len() for s in input_sequence]
    assert len(input_sequence) == len(pitches)
    for curr_note, curr_pitch in zip(input_sequence, pitches):
        
        # if isinstance(curr_note, Note):
        curr_note = id_to_token[curr_note]
        if not curr_note.is_rest and curr_pitch is None:
            print("Error at start0: pitch is None for a note")
            print(curr_note, curr_pitch)
            input()
    
    # create start_end list
    start_end = [(0, note_lens[0], False, id_to_token[input_sequence[0]])] # (onset, offset, to_be_tied_forward, token_id)
    
    for i in range(1, len(note_lens)):
        start_end.append([start_end[-1][1], start_end[-1][1] + note_lens[i], False, id_to_token[input_sequence[i]]])
    # print(start_end)
    m = offset
    for curr_note, curr_pitch in zip(start_end, pitches):
        curr_note = curr_note[3]
        
        # if isinstance(curr_note, Note):
        if not curr_note.is_rest and curr_pitch is None:
            print("Error at start: pitch is None for a note")
            print(curr_note, curr_pitch)
            raise ValueError("Pitch is None for a note")
    i = 0
    results = []
    result_pitches = []
    # print("input pirches: ", pitches)
    while i < len(start_end):
        if start_end[i][1] > m: # split it into two notes
            
            curr = start_end[i]
            if pitches[i] is None and not curr[3].is_rest:
                print("Line 70 error")
                print(curr, offset, beats_per_bar, m)
                # input()
            start_end.pop(i)
            start_end.insert(i, (curr[0], m, True, Note(string=str(curr[3]))))
            start_end.insert(i + 1, (m, curr[1], False, Note(string=str(curr[3]))))
            
            pitches.insert(i, pitches[i])
            
            
        
        results.append(start_end[i])
        result_pitches.append(pitches[i])
        if start_end[i][1] == m:
            results.append(BARLINE_TOKEN)
            result_pitches.append(None)
            m += beats_per_bar
            
        i += 1
    
    for curr_note, curr_pitch in zip(results, result_pitches):
        curr_note = curr_note[3]
        # input()
        if isinstance(curr_note, Note):
            if not curr_note.is_rest and curr_pitch is None:
                print("Error at middle: pitch is None for a note")
                raise ValueError("Pitch is None for a note")
    
    # print("\nresult_pitches:", result_pitches)
    # input()
    result_notes = []
    final_result_pitches = []
    unk = False
    for i, (result, pitch) in enumerate(zip(results, result_pitches)):
        assert len(result_notes) == len(final_result_pitches)
        if result == BARLINE_TOKEN:
            result_notes.append(BARLINE_TOKEN)
            final_result_pitches.append(None)
        else:
            note_result = result[3]
            # note.tied_forward = result[2] if not note.is_rest else False
            notes = dur_to_notes(result[1] - result[0])
            
            if notes is None:                
                # add unknown token
                result_notes.append(UNKNOWN_TOKEN)
                final_result_pitches.append(None)
                unk = True
            else:
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
                    final_result_pitches.append(pitch)
                    
            # note.duration = duration[0]
            # note.dotted = duration[2]
            # note.triplet = duration[3]
    # print("\nfinal:", final_result_pitches)
    # input()
    # make sure any rests are not tied forward
    for i in range(len(result_notes)):
        if isinstance(result_notes[i], Note) and result_notes[i].is_rest:
            result_notes[i].tied_forward = False
            
    for curr_note, curr_pitch in zip(result_notes, final_result_pitches):
        if isinstance(curr_note, Note):
            if not curr_note.is_rest and curr_pitch is None:
                print("Error: pitch is None for a note")
                raise ValueError("Pitch is None for a note")
                # input()
    final_result = [token_to_id[note] if note in token_to_id.keys() else token_to_id[UNKNOWN_TOKEN] for note in result_notes]
    final_result_pitches = [final_result_pitches[i] if final_result[i] != token_to_id[UNKNOWN_TOKEN] else None for i in range(len(final_result))] # add a None wherever there is an unknown token
    
    final_result = [token_to_id[START_OF_SEQUENCE_TOKEN], token_to_id[QUARTER_LENGTHS[beats_per_bar]]] + final_result + [token_to_id[END_OF_SEQUENCE_TOKEN]]
    final_result_pitches = [None, None] + final_result_pitches + [None]
    # if unk:
    #     print("Unknown token found in the sequence")
        # print(f"Result notes: {result_notes}")
        # print(f"Final result: {final_result}")
        # input()
        # input()
    assert len(final_result) == len(final_result_pitches)
    return final_result, final_result_pitches
            
            
def main():
    parser = argparse.ArgumentParser(description="Add barlines to a sequence")
    parser.add_argument("--model_path", type=str, help="Path to language model")
    parser.add_argument("--input_path", type=str, help="Path to input json file")
    parser.add_argument("--note_info_path", type=str, help="Path to note info json file. Used to get the pitches.")
    parser.add_argument("--data_dir", type=str, help="Path to processed data directory")
    parser.add_argument("--output_path", type=str, help="Path to output directory")
    args = parser.parse_args() 
    
    tmp = json.load(open(args.note_info_path, "r"))
    tmp_input = json.load(open(args.input_path, "r"))
    assert len(tmp) == len(tmp_input) # this must hold, otherwise we can't really evaluate later on.
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    # Load
    with no_warnings():
        model = torch.load(args.model_path).to(DEVICE) # supp
    output_dir = "/".join(args.model_path.split("/")[:-1])
    os.makedirs(output_dir, exist_ok=True)
    
    token_to_id, id_to_token, _ = open_processed_data_dir(args.data_dir)
    # print(token_to_id)
    # input()
    flattened_notes, flattened_pitches = flatten_notes(args.input_path, args.note_info_path)
    
        
    # if isinstance(curr_note, Note):
    input_sequence = []
    input_pitches = []
    for note, pitch in zip(flattened_notes, flattened_pitches):
        if note in token_to_id.keys():
            input_sequence.append(token_to_id[note])
            input_pitches.append(pitch)
            
    # print(input_pitches)
    
    candidate_beats_per_bar = [2, 3, 4, 6]
    highest_ll = -float("inf")
    best = None
    best_output_sequence = None
    
    count = 0
    for beats_per_bar in candidate_beats_per_bar:
        candidate_offsets = [i/4 for i in range(int(beats_per_bar * 4))]
        for offset in candidate_offsets:
            count += 1

            # print(f"Trying beats per bar: {beats_per_bar}, offset: {offset}")
            output_sequence, pitches = add_barlines(input_sequence, input_pitches, token_to_id, id_to_token, beats_per_bar, offset)
            # print([id_to_token[k] for k in output_sequence[:20]], pitches[:20])
            # print([(id_to_token[s], p) for s, p in zip(output_sequence[:20], pitches[:20])])
            # input()
            # print(output_sequence)
            output_sequence = torch.Tensor(output_sequence).to(DEVICE).long()
            output, _ = model(output_sequence)
            output = output[:-1, ...]
            # print(f"output shape: {output.shape}")
            # print(f"output_sequence first 10: {output_sequence[:10]}")
            
            # compute the log likelihood
            ll = torch.log_softmax(output, dim=-1)
            ll = torch.gather(ll, 1, output_sequence[1:].unsqueeze(1).to(torch.int64)).squeeze().sum().item()
            if ll > highest_ll:
                highest_ll = ll
                best = (beats_per_bar, offset)
                best_output_sequence = output_sequence
                best_pitches = pitches
    
    # print([id_to_token[b] for b in best_output_sequence.tolist()[:20]])
    # print(best_pitches[:20])
    best_output_sequence = [id_to_token[b] for b in best_output_sequence.tolist()[2:-1]]
    best_pitches = best_pitches[2:-1]
    # print(best_output_sequence)
    # print(best_pitches)
    # print(len(best_output_sequence), len(best_pitches))
    # input()
    best_output = {
        "beats_per_bar": best[0],
        "offset": best[1],
        "log_likelihood": highest_ll, 
        "sequence": best_output_sequence,
        "pitches": best_pitches,
    }
    print(f"Best beats per bar: {best[0]}, offset: {best[1]}, log likelihood: {highest_ll}")
    # save to output directory
    with open(args.output_path, "w") as f:
        f.write(serialize_json(best_output))
        
if __name__ == "__main__":
    main()