import torch
import torch.nn as nn
import argparse
import os
import pickle as pkl
import json
import warnings
from tqdm import tqdm
import sys
from music21 import stream, note, duration

from src.gaussian_decoder.gaussian_decoder import GaussianDecoder
from src.utils import serialize_json, decompose_note_sequence, convert_alignment, open_processed_data_dir
from src.preprocess.ingestion_utils import read_note_text_file
from src.note.Note import Note
from src.const_tokens import *
from src.output_utils import note_pitches_to_xml_no_barlines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rhythm_lstm_path", type=str, required=True, help="Path to the pre-trained rhythm LSTM model .pth file")
    parser.add_argument("--processed_data_dir", type=str, required=True, help="Path to the processed data directory i.e. output of kern_processer.py")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the raw input durations text file")
    parser.add_argument("--tempo", type=float, required=True, help="Tempo in BPM")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width for beam search")
    parser.add_argument("--lambda_param", type=float, default=0.5, help="Weight for combining the rhythm LSTM probabilities and Gaussian channel probabilities (1=only gaussian, 0=only LSTM)")
    parser.add_argument("--sigma", type=float, default=0.1, help="Standard deviation for the Gaussian channel")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    print(args)
    warnings.filterwarnings("ignore")  # Suppress UserWarnings
    tmp = False
    if not tmp:
        event_lengths, pitches, is_note = read_note_text_file(args.input_path)
        rhythm_lstm = torch.load(args.rhythm_lstm_path)

        decoder = GaussianDecoder(
            rhythm_lstm=rhythm_lstm,
            processed_data_dir=args.processed_data_dir,
            beam_width=args.beam_width,
            lambda_param=args.lambda_param,
            sigma=args.sigma
        )
        
        best_sequences, best_scores, best_pitches = decoder(
            input_durations=event_lengths,
            is_note=is_note,
            tempo=args.tempo,
            pitches=pitches
        )
        # first assert that, wherever best_pithces is None, the sequence is a rest. Also, wherever best_pitches is not None, the sequence is a note.
        for seq, curr_pitches in zip(best_sequences, best_pitches):
            for tok, pitch in zip(seq, curr_pitches):
                if pitch is None:
                    assert type(tok) == str or tok.is_rest, "Mismatch between predicted pitch and token type (rest vs note)"
                else:
                    assert not tok.is_rest, "Mismatch between predicted pitch and token type (rest vs note)"
        # save output
        results = {i: {"sequence": seq, "score": score, "pitches": curr_pitches} for i, (seq, score, curr_pitches) in enumerate(zip(best_sequences, best_scores, best_pitches))}
        
        os.makedirs(args.output_dir, exist_ok=True)
    # output_path = os.path.join(args.output_dir, "decoded_sequences.json")
    # actually, save each sequence in separate json files
    # with open(output_path, 'w') as f:
    #     f.write(serialize_json(results))
    # 
    
    # for now just load the saved
    if tmp:
        loaded = json.load(open("output/gaussian_decoder/test_tempo_74.81/decoded_sequence_0.json", 'r'))["sequence"]
        # evaluate each tuple
        
        
        
        evaled_loaded = [eval(t) for t in loaded]
        notes = [eval(t[0]) if not (t[0][0] == "<") else t[0] for t in evaled_loaded][1:-1]  # remove start and end tokens
        pitches = [t[1] for t in evaled_loaded][1:-1]
        
        note_pitches_to_xml_no_barlines(notes, pitches, "output/gaussian_decoder/test_tempo_74.81/decoded_sequence_0.ly", num_events_per_measure=16)
    
    if not tmp:
        for i, (seq, score, curr_pitches) in enumerate(zip(best_sequences,
                                                            best_scores,
                                                            best_pitches)):
            seq_output_path = os.path.join(args.output_dir, f"decoded_sequence_{i}.json")
            seq_result = {"sequence": [str((str(s), pitch)) for s, pitch in zip(seq, curr_pitches)], "score": score}
            with open(seq_output_path, 'w') as f:
                json.dump(seq_result, f, indent=4)    
                    
            # save mxl files, no barlines for now
            seq = seq[1:-1]  # remove start and end tokens
            curr_pitches = curr_pitches[1:-1]
            mxl_output_path = os.path.join(args.output_dir, f"decoded_sequence_{i}.ly")
            note_pitches_to_xml_no_barlines(seq, curr_pitches, mxl_output_path, num_events_per_measure=16)
        
    