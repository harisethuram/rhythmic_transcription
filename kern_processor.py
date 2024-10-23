# output: tokenized_dataset.pkl, token_to_id.pkl, id_to_token.pkl

from music21 import converter, note
import music21
import argparse
import os
from tqdm import tqdm
import pickle as pkl
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess.tokenizer_utils import get_rhythms_and_expressions, serialize_json, get_tuple, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts a directory of kern files into a list of tokens.")
    parser.add_argument("--input_dirs", type=str, help="Comma-separated list of directories containing kern files.")
    parser.add_argument("--output_dir", type=str, help="The file to write the tokenized output to.")
    parser.add_argument("--train_split", type=float, default=0.9, help="The proportion of the data to use for training.")
    parser.add_argument("--want_barlines", action="store_true", help="Whether or not to include barlines in the output.")
    parser.add_argument("--no_expressions", action="store_true", help="Whether or not to include expressions in the output.")
    parser.add_argument("--test_limit", type=int, default=100000, help="The number of files to process for testing")
    args = parser.parse_args()

    # get the rhyths and expressions for every part for every file in the input directory
    print("Getting rhythms and expressions for all parts in all files...")
    all_rhythms_and_expressions = {}
    for dataset in args.input_dirs.split(","):
        for file in tqdm(os.listdir(dataset)):
            if file.split(".")[-1] != "krn":
                continue
            
            parts = converter.parse(os.path.join(dataset, file)).parts
            all_rhythms_and_expressions[os.path.join(dataset, file)] = {}
            for i, part in enumerate(parts):
                all_rhythms_and_expressions[os.path.join(dataset, file)][i] = get_rhythms_and_expressions(part, args.want_barlines, args.no_expressions)
    
    # get all unique note tokens
    note_tokens = []
    for all_parts in all_rhythms_and_expressions.values():
        for part in all_parts.values():
            for note in part:
                note_tokens.append((note["duration"], note["dotted"], note["triplet"], note["fermata"], note["staccato"], note["tied_forward"], note["is_rest"]))
    unique_note_tokens = list(set(note_tokens))
    
    token_to_id, id_to_token = tokenizer(args.want_barlines, args.no_expressions)
    max_id = max(token_to_id.values())
    print(token_to_id)
    count = max_id + 1
    print("Number of unique note tokens:", len(token_to_id.keys()))
    
    print("Tokenizing rhythms and expressions...")
    split_names = ["train", "val"]
    all_rhythms_and_expressions_tokenized = {}
    for piece_name, parts in all_rhythms_and_expressions.items():
        all_rhythms_and_expressions_tokenized[piece_name] = {}
        for part, notes in parts.items():
            rhythms_and_expressions_tokenized = []
            for note in notes:
                tup = get_tuple(note["duration"], note["dotted"], note["triplet"], note["fermata"], note["staccato"], note["tied_forward"], note["is_rest"])
                if tup not in token_to_id.keys():
                    print("Token not found in dictionary:", tup, "in piece", piece_name, "part", part)
                    token_to_id[tup] = count
                    id_to_token[count] = tup
                    count += 1
                rhythms_and_expressions_tokenized.append(token_to_id[tup])
            all_rhythms_and_expressions_tokenized[piece_name][part] = rhythms_and_expressions_tokenized
        
        all_rhythms_and_expressions_tokenized[piece_name]["split"] = split_names[int(random.random() > args.train_split)]
    print("Number of unique note tokens after tokenization:", len(token_to_id.keys()))
    # save the tokenized output and dictionaries as pickle files
    print("Writing tokenized output to file...")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "tokenized_dataset.json"), "w") as f:
        f.write(serialize_json(all_rhythms_and_expressions_tokenized))
        
    with open(os.path.join(args.output_dir, "token_to_id.pkl"), "wb") as f:
        pkl.dump(token_to_id, f)
        
    with open(os.path.join(args.output_dir, "id_to_token.pkl"), "wb") as f:
        pkl.dump(id_to_token, f)
