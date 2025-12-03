# takes in directories to kern files, creates a tokenizer and tokenizes them.
# output: tokenized_dataset.pkl, token_to_id.pkl, id_to_token.pkl

from music21 import converter, note
import music21
import argparse
import os
from tqdm import tqdm
import pickle as pkl
import sys
import random
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess.tokenizer_utils import get_rhythms_and_expressions, get_base_tokenizer_dicts, tokenize
from const_tokens import *
from src.utils import serialize_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts a directory of kern files into a list of tokens.")
    parser.add_argument("--input_dirs", type=str, help="Comma-separated list of directories containing kern files.")
    parser.add_argument("--output_dir", type=str, help="The file to write the tokenized output to.")
    parser.add_argument("--train_split", type=float, default=0.9, help="The proportion of the data to use for training.")
    parser.add_argument("--want_barlines", action="store_true", help="Include barlines in the output.")
    parser.add_argument("--no_expressions", action="store_true", help="Don't include expressions in the output.")
    parser.add_argument("--test_limit", type=int, default=100000, help="The number of files to process for testing")
    parser.add_argument("--debug", action="store_true", help="Whether or not to print debug information.")
    parser.add_argument("--unk_threshold", type=int, default=100, help="The frequency threshold below which a token is considered unknown.")
    parser.add_argument("--want_measure_lengths", action="store_true", help="Whether or not to include measure lengths in the output.")
    parser.add_argument("--want_elucidation", action="store_true", help="Whether or not to have every single note elucidated in the tokenizer.")
    args = parser.parse_args()
    
    random.seed(0)
    os.makedirs(args.output_dir, exist_ok=True)
    print(args)
    # get the rhyths and expressions for every part for every file in the input directory
    print("Getting rhythms and expressions for all parts in all files...")
    all_rhythms_and_expressions = {}
    total_num_parts = 0
    num_useless_parts = 0
    
    num_corrupt_files = 0
    total_num_files = 0
    num_corrupt_parts = 0
    # total_num_parts = 0
    for dataset in args.input_dirs.split(","):
        print("Processing", dataset)
        for file in tqdm(os.listdir(dataset)):
            
            if file.split(".")[-1] != "krn":
                continue
            total_num_files += 1
            try:
                parts = converter.parse(os.path.join(dataset, file)).parts
            except Exception as e:
                num_corrupt_files += 1
                continue
            all_rhythms_and_expressions[os.path.join(dataset, file)] = {}
            # print("path:", os.path.join(dataset, file))
            part_counter = 0
            for i, part in enumerate(parts):
                corrupted = False
                try:
                    tmp_parts, polyphonic_bars = get_rhythms_and_expressions(part, want_barlines=args.want_barlines, no_expressions=args.no_expressions, want_measure_lengths=args.want_measure_lengths) # a list of lists of notes
                except Exception as e:
                    tmp_parts = []
                    polyphonic_bars = None
                    print(f"Error in file {os.path.join(dataset, file)} part {i}: {e}")
                    corrupted = True
                    num_corrupt_parts += 1
                    
                for tmp_part in tmp_parts:
                    all_rhythms_and_expressions[os.path.join(dataset, file)][part_counter] = tmp_part
                    part_counter += 1
                
                # all_rhythms_and_expressions[os.path.join(dataset, file)][f"polyphonic_bars_{part_counter}"] = str(polyphonic_bars)
                # except Exception as e:
                # tmp=None
                # print(f"Error in file {os.path.join(dataset, file)} part {i}: {e}")
                if len(tmp_parts) == 0 and not corrupted:
                    num_useless_parts += 1
                # else:
                #     print(f"Skipping file {os.path.join(dataset, file)} part {i}")
                total_num_parts += 1
                # print(tmp_parts)
                # for r in tmp_parts:
                #     print("*************************")
                #     start = r[:20]
                #     end = r[-20:]
                #     for j in start:
                #         print(j)
                #     print("...")
                #     for j in end:
                #         print(j)
                # print(f"num spliced parts for {os.path.join(dataset, file)} - {i}:", len(tmp_parts))
                # if len(tmp_parts) > 0 and "offering-001" in os.path.join(dataset, file):
                #     print(tmp_parts[0])
                #     print("path again:", os.path.join(dataset, file))
                # input()

                # input()
    print(f"Number of useless parts: {num_useless_parts}/{total_num_parts} ({num_useless_parts/total_num_parts*100:.2f}%)")
    print(f"Number of corrupt parts: {num_corrupt_parts}/{total_num_parts} ({num_corrupt_parts/total_num_parts*100:.2f}%)")
    print(f"Number of corrupted files: {num_corrupt_files}/{total_num_files} ({num_corrupt_files/total_num_files*100:.2f}%)")
    # remove all files with no useful parts
    
    # get all unique note tokens
    
    unique_note_tokens = set()
    for all_parts in all_rhythms_and_expressions.values():
        for part in all_parts.values():
            for note in part:
                unique_note_tokens.add(note)
    
    token_to_id, id_to_token = get_base_tokenizer_dicts(args.want_barlines, args.no_expressions, args.want_elucidation)
    print("Base tokenizer dicts created.")
    # print(id_to_token)
    
    # train stats
    
    train_total_num_notes = 0
    
    train_frequencies = {}
    train_unk_tokens = []
    
    print("Number of unique note tokens:", len(token_to_id.keys()))
    
    print("Tokenizing rhythms and expressions...")
    split_names = ["train", "val"]
    all_rhythms_and_expressions_tokenized = {}
    
    # assign splits to each piece and get frequencies for each token in train split and basically build the tokenizer
    for piece_name, parts in all_rhythms_and_expressions.items(): # for each piece
        if args.debug:
            print(f"Piece: {piece_name}")
            
        all_rhythms_and_expressions_tokenized[piece_name] = {}
        curr_split = split_names[int(random.random() > args.train_split)]
        all_rhythms_and_expressions_tokenized[piece_name]["split"] = curr_split
        
        if curr_split == "train":
            for part, notes in parts.items():
                if args.debug:
                    print(f"Parsing part {part} of piece {piece_name}...")
                    print(notes)
                    input()

                for note in notes:
                    train_frequencies[note] = train_frequencies.get(note, 0) + 1
                    
    
    # remove tokens with frequency below unk_threshold
    if args.want_elucidation:
        # if we want elucidation, we don't filter out any tokens
        filtered_train_frequencies = train_frequencies
    else:
        filtered_train_frequencies = {k: v for k, v in train_frequencies.items() if v > args.unk_threshold}
        print(f"Number of tokens with frequency below threshold ({args.unk_threshold}):", len(train_frequencies) - len(filtered_train_frequencies), "out of", len(train_frequencies))
        filtered_train_frequencies[UNKNOWN_TOKEN] = sum([v for k, v in train_frequencies.items() if v <= args.unk_threshold])
    
    max_id = max(token_to_id.values())
    token_id_count = max_id + 1
    
    for token in filtered_train_frequencies.keys():
        if token not in token_to_id.keys():
            # token_to_id[token] = token_id_count
            token_to_id[token] = token_id_count
            id_to_token[token_id_count] = token
            token_id_count += 1
            
    if args.debug:
        print("token_to_id:")
        print(token_to_id)
        print("id_to_token:")
        print(id_to_token)
        input()
    # plot the frequencies of the tokens along with the threshold
    freq_list = [(k, v) for k, v in train_frequencies.items()]
    freq_list.sort(key=lambda x: x[1], reverse=True)
    plt.plot([x[1] for x in freq_list], label="Note frequencies")
    plt.axhline(args.unk_threshold, color="red", label="UNK Threshold")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title("Rank vs Train Frequency of Note Tokens")
    plt.yscale("log")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "note_train_frequencies_threshold.png"))    
        
    # now actually tokenize the rhythms and expressions
    num_train_unks = 0
    num_train_tokens = 0
    
    for piece_name, parts in all_rhythms_and_expressions.items(): # for each piece
        if args.debug:
            print(f"Piece: {piece_name}")
        curr_split = all_rhythms_and_expressions_tokenized[piece_name]["split"]
        for part, notes in parts.items():
            if args.debug:
                print(f"Parsing part {part} of piece {piece_name}...")
                print(notes)
                input()
            rhythms_and_expressions_tokenized = []
            if curr_split == "train":
                num_train_tokens += len(notes)
                
            for note in notes:
                # note = str(note)
                if note in token_to_id.keys():
                    rhythms_and_expressions_tokenized.append(token_to_id[note])
                else:
                    rhythms_and_expressions_tokenized.append(token_to_id[UNKNOWN_TOKEN])
                    train_unk_tokens.append(note)
                    if curr_split == "train":
                        num_train_unks += 1
                    
            all_rhythms_and_expressions_tokenized[piece_name][part] = rhythms_and_expressions_tokenized
            # all_rhythms_and_expressions_tokenized[piece_name]["polyphonic_bars"] = str(parts["polyphonic_bars"])
        # add the polyphonic bars to the tokenized output
        # for part in parts.keys():
        #     if "polyphonic_bars" in str(part):
        #         all_rhythms_and_expressions_tokenized[piece_name][part] = parts[part]
            
    all_rhythms_and_expressions_tokenized = {k: v for k, v in all_rhythms_and_expressions_tokenized.items() if len(v) > 1}                
                
    assert len(token_to_id.keys()) == len(id_to_token.keys())
    print(f"Number of token ids: {len(token_to_id.keys())}")
    print("\nTrain set stats:")
    print(f"Number of unknown tokens: {num_train_unks}/{num_train_tokens} ({num_train_unks/num_train_tokens*100:.2f}%)")
    print(f"Median note frequency: {sorted(filtered_train_frequencies.values())[len(filtered_train_frequencies)//2]}")
    print(f"Mean note frequency: {sum(filtered_train_frequencies.values())/len(token_to_id.keys())}")
    print(f"Max note frequency: {max(filtered_train_frequencies.values())}")
        
    # save the tokenized output and dictionaries as pickle files
    print("Writing tokenized output to file...")
    
    with open(os.path.join(args.output_dir, "tokenized_dataset.json"), "w") as f:
        f.write(serialize_json(all_rhythms_and_expressions_tokenized))
        
    with open(os.path.join(args.output_dir, "token_to_id.pkl"), "wb") as f:
        pkl.dump(token_to_id, f)
        
    with open(os.path.join(args.output_dir, "id_to_token.pkl"), "wb") as f:
        pkl.dump(id_to_token, f)
    
    # also save id_to_token as json file
    with open(os.path.join(args.output_dir, "id_to_token.json"), "w") as f:
        id_to_token_str = {k: str(v) for k, v in id_to_token.items()}
        f.write(serialize_json(id_to_token_str))
        
    with open(os.path.join(args.output_dir, "token_to_id.json"), "w") as f:
        # first convert the token_to_id dict to a dict with string keys
        token_to_id_str = {str(k): v for k, v in token_to_id.items()}
        f.write(serialize_json(token_to_id_str))
        
    # also save train_frequencies as json file
    with open(os.path.join(args.output_dir, "train_frequencies.json"), "w") as f:
        f.write(serialize_json(filtered_train_frequencies))
    
    metadata = {
        "num_train_unks": num_train_unks,
        "num_train_tokens": num_train_tokens,
        "want_barlines": args.want_barlines,
        "no_expressions": args.no_expressions,
        "train_split": args.train_split,
        "unk_threshold": args.unk_threshold,
    }
   
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        f.write(serialize_json(metadata))
