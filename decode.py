# actually does the decoding

import torch
import torch.nn as nn
import music21
import argparse
import os
import pickle as pkl
import json
import warnings
from tqdm import tqdm
import sys

from src.decoder.decoder import Decoder
from src.utils import serialize_json, decompose_note_sequence, convert_alignment
from src.note.Note import Note
from const_tokens import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language_model_path", type=str, required=True, help="Path to the language model")
    parser.add_argument("--channel_model_path", type=str, default=None, help="Path to the channel model")
    parser.add_argument("--processed_data_dir", type=str, required=True, help="Path to the processed data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--note_info_path", type=str, required=True, help="Path to the note info json file. This file contains an array consisting of length 3 (or 4) arrays: [quantized_length, note_portion, rest_portion, pitch (optional)]")
    parser.add_argument("--decode_method", type=str, default="greedy", help="Decoding method to use")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width for beam search")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling") # TODO: get rid of this
    parser.add_argument("--base_value", type=float, default=1.0, help="What length a quarter note corresponds to")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--score_path", type=str, default=None, help="Path to the tokenized json file, only required if eval is true")
    parser.add_argument("--score_part_id", type=int, default=None, help="Part ID of the score, only required if eval is true")
    parser.add_argument("--alignment_path", type=str, default=None, help="Path to the alignment json file consisting of alignment between quantized performance and score, only required if eval is true")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    print(args)
    warnings.filterwarnings("ignore")  # Suppress UserWarnings
    language_model = torch.load(args.language_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    language_model.to(device)
    if args.channel_model_path:
        channel_model = torch.load(args.channel_model_path)
    else:
        channel_model = None
    
    with open(os.path.join(args.processed_data_dir, "token_to_id.pkl"), "rb") as f:
        token_to_id = pkl.load(f)
    with open(os.path.join(args.processed_data_dir, "id_to_token.pkl"), "rb") as f:
        id_to_token = pkl.load(f)
    with open(args.note_info_path, "r") as f:
        note_info = json.load(f)
    
    id_to_token = {tok_id: (Note(tple=token) if type(token) == tuple else token) for tok_id, token in id_to_token.items()}
    token_to_id = {(Note(tple=token) if type(token) == tuple else token): tok_id for token, tok_id in token_to_id.items()}
      
    decoder = Decoder(language_model, channel_model, token_to_id, id_to_token, args.beam_width, args.temperature, args.base_value)
    

    output, detokenized_output = decoder.decode(note_info=note_info, decode_method=args.decode_method, flatten=False, debug=args.debug)

    os.makedirs(args.output_dir, exist_ok=True)
    
    # detokenized_output = [list(x) if type(x) != str else [x] for x in detokenized_output]
    
    with open(os.path.join(args.output_dir, "output.json"), "w") as f:
        f.write(serialize_json(detokenized_output))
        
    # evaluation
    if args.eval:
        if args.score_path is None or not os.path.exists(args.score_path):
            raise ValueError("Score path is required for evaluation")
        if args.score_part_id is None:
            raise ValueError("Score part ID is required for evaluation")
        if args.alignment_path is None or not os.path.exists(args.alignment_path):
            raise ValueError("Alignment path is required for evaluation")
        
        with open(args.score_path, "r") as f:
            score_token_ids = json.load(f)[str(args.score_part_id)]
        score_token_ids = [int(x) for x in score_token_ids]
        # print("score ids:", score_token_ids)
        _, score_notes = decompose_note_sequence(score_token_ids, token_to_id, id_to_token)
        
        with open(args.alignment_path, "r") as f:
            alignment = json.load(f)
        if alignment[-1][1] >= len(detokenized_output):
            print("error: alignment length is greater than detokenized output length, must be missing some notes")
            sys.exit(1)
        # alignment_dict = {}
        # for score_index, performance_index in alignment:
        #     alignment_dict[performance_index] = alignment_dict.get(performance_index, []) + [score_index]
        alignment_dict = convert_alignment(alignment)
        # print(alignment_dict)
        results = {}
        
        correct = []
        length_se = []
        correct_given_length_se_0 = []
        total = len(alignment_dict)
        # print("score notes:", score_notes, len(score_notes))
        # print("length:", len(score_notes), len(note_info), len(detokenized_output))
        for score, perf in tqdm(alignment_dict.items()):
            # print(score, perf)
            # print(detokenized_output[perf[0]])
            # print(score_notes[score])
            # print(score, )
            curr_score_notes = []
            curr_score_rests = []
            for score_index in score:
                curr_score_notes += score_notes[score_index][0]
                curr_score_rests += score_notes[score_index][1]
                
            curr_perf_notes = []
            curr_perf_rests = []
            for perf_index in perf:
                curr_perf_notes += detokenized_output[perf_index][0]
                curr_perf_rests += detokenized_output[perf_index][1]
            # print("score:", curr_score_notes)
            # print("rests:", curr_score_rests)
            # print("notes:", curr_perf_notes)   
            # print("rests:", curr_perf_rests)
            # print("type:", type(curr_perf_notes[0]))
            
            # binary correct?
            if curr_score_notes == curr_perf_notes and curr_score_rests == curr_perf_rests:
                correct.append(1)
            else:
                correct.append(0)
            # print(correct[-1])  
            # input()
            # length squared error
            curr_score_note_length = sum([note.get_len() for note in curr_score_notes])
            curr_perf_note_length = sum([note.get_len() for note in curr_perf_notes])
            curr_score_rest_length = sum([rest.get_len() for rest in curr_score_rests])
            curr_perf_rest_length = sum([rest.get_len() for rest in curr_perf_rests])
            
            length_se.append((curr_score_note_length - curr_perf_note_length) ** 2 + (curr_score_rest_length - curr_perf_rest_length) ** 2)
            
            # length squared error given binary correct
            if length_se[-1] == 0:
                correct_given_length_se_0.append(correct[-1])
            else:
                correct_given_length_se_0.append(None)
        
        # aggregate results
        results["total_binary_accuracy"] = float(float(sum(correct)) / float(total))
        results["total_length_se"] = float(float(sum(length_se)) / float(total))
        results["total_correct_given_length_se_0"] = float(float(sum([x for x in correct_given_length_se_0 if x is not None])) / float(sum([1 for x in correct_given_length_se_0 if x is not None]) + 1e-6))
        
        results["binary_correct"] = correct
        results["length_se"] = length_se
        results["correct_given_length_se_0"] = correct_given_length_se_0
        
        print(results)
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            f.write(serialize_json(results))