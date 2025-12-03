# evaluates the output of the decode algorithm. 

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
from src.utils import serialize_json, decompose_note_sequence, convert_alignment, open_processed_data_dir
from src.note.Note import Note
from src.const_tokens import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_path", type=str, required=True, help="Path to the tokenized json file, only required if eval is true")
    parser.add_argument("--score_part_id", type=int, required=True, help="Part ID of the score, only required if eval is true")
    parser.add_argument("--transcription_path", type=str, required=True, help="Path to the json file output of the decoder")
    parser.add_argument("--transcription_part_id", type=int, required=True, help="Part ID of the transcription, only required if eval is true")
    parser.add_argument("--alignment_path", type=str, required=True, help="Path to the alignment json file consisting of alignment between quantized performance and score, only required if eval is true")
    parser.add_argument("--output_dir", type=str, required=True, help="output_dir")
    parser.add_argument("--processed_data_dir", type=str, required=True, help="path to tokenizer dictionary, etc...")
    
    args = parser.parse_args()
    print(args)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # we need to get an alignment for the score to performance transcription, but that depends on the score and performance
    # to get this alignment, should we do onset-onset length? i think so...
    token_to_id, id_to_token, metadata = open_processed_data_dir(args.processed_data_dir)
    
    with open(args.transcription_path, "r") as f:
        raw_output = json.load(f)[str(args.transcription_part_id)]
    raw_output = [int(token) for token in raw_output]
    
    output, detokenized_output = decompose_note_sequence(raw_output, token_to_id, id_to_token)
    
    
    # evaluation
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