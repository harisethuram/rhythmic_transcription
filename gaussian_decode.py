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
    parser.add_argument("--tempos", type=str, required=True, help="Comma separated tempos in BPM")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width for beam search")
    parser.add_argument("--lambda_param", type=float, default=0.5, help="Weight for combining the rhythm LSTM probabilities and Gaussian channel probabilities (1=only gaussian, 0=only LSTM)")
    parser.add_argument("--sigma", type=float, default=0.1, help="Standard deviation for the Gaussian channel")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    print(args)
    warnings.filterwarnings("ignore")  # Suppress UserWarnings
    # if output dir exists, and folder isn't empty, exit
    if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)) > 0 and os.path.exists(os.path.join(args.output_dir, "decoded_sequence_0.json")):
        print(f"Output directory {args.output_dir} already exists. Exiting to avoid overwriting.")
        sys.exit(1)
    event_lengths, pitches, is_note = read_note_text_file(args.input_path)
    rhythm_lstm = torch.load(args.rhythm_lstm_path)

    decoder = GaussianDecoder(
        rhythm_lstm=rhythm_lstm,
        processed_data_dir=args.processed_data_dir,
        beam_width=args.beam_width,
        lambda_param=args.lambda_param,
        sigma=args.sigma
    )
    
    tempos = [float(t) for t in args.tempos.split(",")]
    all_best_sequences = []
    all_best_scores = []
    all_best_pitches = []
    all_best_tempos = []
    for tempo in tempos:
        print(f"Decoding for tempo: {tempo}")
        curr_best_sequences, curr_best_scores, curr_best_pitches = decoder(
            input_durations=event_lengths,
            is_note=is_note,
            tempo=tempo,
            pitches=pitches
        )
        
        all_best_sequences.extend(curr_best_sequences)
        all_best_scores.extend(curr_best_scores)
        all_best_pitches.extend(curr_best_pitches)
        all_best_tempos.extend([tempo]*len(curr_best_sequences))
        
    best_sequences = all_best_sequences
    best_scores = all_best_scores
    best_pitches = all_best_pitches
    best_tempos = all_best_tempos
    
    # first assert that, wherever best_pithces is None, the sequence is a rest. Also, wherever best_pitches is not None, the sequence is a note.
    for seq, curr_pitches in zip(best_sequences, best_pitches):
        for tok, pitch in zip(seq, curr_pitches):
            if pitch is None:
                assert type(tok) == str or tok.is_rest, "Mismatch between predicted pitch and token type (rest vs note)"
            else:
                assert not tok.is_rest, "Mismatch between predicted pitch and token type (rest vs note)"
                    
    # get top beam_width sequences
    best_idxs = sorted(range(len(best_scores)), key=lambda i: best_scores[i], reverse=True)[:args.beam_width]
    best_sequences = [best_sequences[i] for i in best_idxs]
    best_scores = [best_scores[i] for i in best_idxs]
    best_pitches = [best_pitches[i] for i in best_idxs]
    best_tempos = [best_tempos[i] for i in best_idxs]
    print(best_scores)
    print(best_tempos)
    print(best_idxs)
    # save output
    results = {i: {"sequence": seq, "score": score, "pitches": curr_pitches, "tempo": curr_tempo} for i, (seq, score, curr_pitches, curr_tempo) in enumerate(zip(best_sequences, best_scores, best_pitches, best_tempos))}
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # for now just load the saved
    for i, (seq, score, curr_pitches, curr_tempo) in enumerate(zip(best_sequences,
                                                        best_scores,
                                                        best_pitches,
                                                        best_tempos)):
        seq_output_path = os.path.join(args.output_dir, f"decoded_sequence_{i}.json")
        seq_result = {"sequence": [str((str(s), pitch)) for s, pitch in zip(seq, curr_pitches)], "score": score, "tempo": curr_tempo}
        with open(seq_output_path, 'w') as f:
            json.dump(seq_result, f, indent=4)    
                
        # save mxl files, no barlines for now
        seq = seq[1:-1]  # remove start and end tokens
        curr_pitches = curr_pitches[1:-1]
        mxl_output_path = os.path.join(args.output_dir, f"decoded_sequence_{i}.ly")
        note_pitches_to_xml_no_barlines(seq, curr_pitches, mxl_output_path, num_events_per_measure=16)