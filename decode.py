# actually does the decoding

import torch
import torch.nn as nn
import music21
import argparse
import os
import pickle as pkl
import json

from src.decoder.decoder import Decoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the language model")
    parser.add_argument("--channel_path", type=str, default=None, help="Path to the channel model")
    parser.add_argument("--processed_data_dir", type=str, required=True, help="Path to the processed data directory")
    parser.add_argument("--note_info_path", type=str, required=True, help="Path to the note info json file. This file contains an array consisting of length 3 (or 4) arrays: [quantized_length, note_portion, rest_portion, pitch (optional)]")
    parser.add_argument("--decode_method", type=str, default="greedy", help="Decoding method to use")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width for beam search")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    
    args = parser.parse_args()
    
    language_model = torch.load(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    language_model.to(device)
    if args.channel_path:
        channel_model = torch.load(args.channel_path)
    else:
        channel_model = None
    
    with open(os.path.join(args.processed_data_dir, "token_to_id.pkl"), "rb") as f:
        token_to_id = pkl.load(f)
    with open(os.path.join(args.processed_data_dir, "id_to_token.pkl"), "rb") as f:
        id_to_token = pkl.load(f)
    with open(args.note_info_path, "r") as f:
        note_info = json.load(f)
    decoder = Decoder(language_model, channel_model, token_to_id, args.beam_width, args.temperature)
    
    output = decoder.decode(note_info, args.decode_method)
    detokenized_output = [id_to_token[token_id] for token_id in output]
    print(detokenized_output)    