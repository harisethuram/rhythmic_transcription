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
from src.preprocess.tokenizer_utils import get_rhythms_and_expressions, serialize_json



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts a directory of kern files into a list of tokens.")
    parser.add_argument("--input_dir", type=str, help="The directory containing the kern files.")
    parser.add_argument("--output_dir", type=str, help="The file to write the tokenized output to.")
    parser.add_argument("--train_split", type=float, default=0.9, help="The proportion of the data to use for training.")
    parser.add_argument("--want_barlines", action="store_true", help="Whether or not to include barlines in the output.")
    parser.add_argument("--test_limit", type=int, default=100000, help="The number of files to process for testing")
    args = parser.parse_args()

    # get the rhyths and expressions for every part for every file in the input directory
    print("Getting rhythms and expressions for all parts in all files...")
    all_rhythms_and_expressions = {}
    count = 0
    for file in tqdm(os.listdir(args.input_dir)):
        if count > args.test_limit:
            break
        count += 1
        
        parts = converter.parse(os.path.join(args.input_dir, file)).parts
        all_rhythms_and_expressions[file] = {}
        for i, part in enumerate(parts):
            all_rhythms_and_expressions[file][i] = get_rhythms_and_expressions(part, args.want_barlines)
    
    # get all unique note tokens
    note_tokens = []
    for all_parts in all_rhythms_and_expressions.values():
        for part in all_parts.values():
            for note in part:
                note_tokens.append((note["duration"], note["dotted"], note["triplet"], note["fermata"], note["staccato"], note["tied_forward"], note["is_rest"]))

    unique_note_tokens = list(set(note_tokens))

    # assign token ids to each unique note token
    token_to_id = {}
    id_to_token = {}
    # barline
    
    # pad token
    token_to_id[(-1, False, False, False, False, False, False)] = 0
    id_to_token[0] = (-1, False, False, False, False, False, False)
    offset = 1
    
    if args.want_barlines:
        token_to_id[(0, False, False, False, False, False, False)] = 1
        id_to_token[1] = (0, False, False, False, False, False, False)
        offset += 1
    
    for i, token in enumerate(unique_note_tokens):
        token_to_id[token] = i + offset
        id_to_token[i + offset] = token
    
    print("Number of unique note tokens:", len(token_to_id.keys()))
    
    print("Tokenizing rhythms and expressions...")
    split_names = ["train", "val"]
    all_rhythms_and_expressions_tokenized = {}
    for piece_name, parts in all_rhythms_and_expressions.items():
        all_rhythms_and_expressions_tokenized[piece_name] = {}
        for part, notes in parts.items():
            rhythms_and_expressions_tokenized = []
            for note in notes:
                rhythms_and_expressions_tokenized.append(token_to_id[(note["duration"], note["dotted"], note["triplet"], note["fermata"], note["staccato"], note["tied_forward"], note["is_rest"])])
            all_rhythms_and_expressions_tokenized[piece_name][part] = rhythms_and_expressions_tokenized
        
        all_rhythms_and_expressions_tokenized[piece_name]["split"] = split_names[int(random.random() > args.train_split)]
    # print(all_rhythms_and_expressions_tokenized)
    # save the tokenized output and dictionaries as pickle files
    print("Writing tokenized output to file...")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "tokenized_dataset.json"), "w") as f:
        f.write(serialize_json(all_rhythms_and_expressions_tokenized))
        
    with open(os.path.join(args.output_dir, "token_to_id.pkl"), "wb") as f:
        pkl.dump(token_to_id, f)
        
    with open(os.path.join(args.output_dir, "id_to_token.pkl"), "wb") as f:
        pkl.dump(id_to_token, f)
