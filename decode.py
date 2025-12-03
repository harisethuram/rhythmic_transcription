# Decodes the output of the piecewise linear fit. 

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
    parser.add_argument("--language_model_path", type=str, required=True, help="Path to the language model .pth file")
    parser.add_argument("--channel_model_path", type=str, default=None, help="Path to the channel model .pth file")
    parser.add_argument("--processed_data_dir", type=str, required=True, help="Path to the processed data directory i.e. output of kern_processer.py")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--note_info_path", type=str, required=True, help="Path to the note info json file. This file contains an array consisting of length 3 (or 4) arrays: [quantized_length, note_portion, rest_portion, pitch (optional)]")
    parser.add_argument("--decode_method", type=str, default="greedy", help="Decoding method to use, either greedy or beam search")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width for beam search")
    # parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling") # TODO: get rid of this
    parser.add_argument("--base_value", type=float, default=1.0, help="What length a quarter note corresponds to")
    parser.add_argument("--want_mixing", type=str, default="False", help="Whether to run mixing or not. If set to 'True', the output will be mixed with the channel model. If set to 'False', the output will not be mixed.")
    # parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    # parser.add_argument("--score_path", type=str, default=None, help="Path to the tokenized json file, only required if eval is true")
    # parser.add_argument("--score_part_id", type=int, default=None, help="Part ID of the score, only required if eval is true")
    # parser.add_argument("--alignment_path", type=str, default=None, help="Path to the alignment json file consisting of alignment between quantized performance and score, only required if eval is true")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    print(args)
    want_mixing = args.want_mixing.lower() == "true"
    warnings.filterwarnings("ignore")  # Suppress UserWarnings
    language_model = torch.load(args.language_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    language_model.to(device)
    if args.channel_model_path:
        channel_model = torch.load(args.channel_model_path)
    else:
        channel_model = None
    
    # with open(os.path.join(args.processed_data_dir, "token_to_id.pkl"), "rb") as f:
    #     token_to_id = pkl.load(f)
    # with open(os.path.join(args.processed_data_dir, "id_to_token.pkl"), "rb") as f:
    #     id_to_token = pkl.load(f)
    token_to_id, id_to_token, _ = open_processed_data_dir(args.processed_data_dir)
    with open(args.note_info_path, "r") as f:
        note_info = json.load(f)
    
    # id_to_token = {tok_id: (Note(tple=token) if type(token) == tuple else token) for tok_id, token in id_to_token.items()}
    # token_to_id = {(Note(tple=token) if type(token) == tuple else token): tok_id for token, tok_id in token_to_id.items()}
    
    alpha = 1/20 if want_mixing else 1
    
    decoder = Decoder(language_model, channel_model, token_to_id, id_to_token, args.beam_width, args.base_value, alpha=alpha)
    

    output, detokenized_output = decoder.decode(note_info=note_info, decode_method=args.decode_method, flatten=False, debug=args.debug)

    os.makedirs(args.output_dir, exist_ok=True)
    
    # detokenized_output = [list(x) if type(x) != str else [x] for x in detokenized_output]
    
    with open(os.path.join(args.output_dir, "output.json"), "w") as f:
        f.write(serialize_json(detokenized_output))