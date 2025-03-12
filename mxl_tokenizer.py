from music21 import converter, note
import music21
import argparse
import os
import matplotlib.pyplot as plt
import json
import random

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess.tokenizer_utils import get_rhythms_and_expressions
from const_tokens import *
from src.utils import serialize_json, open_processed_data_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tokenizes a musicxml file")
    parser.add_argument("--input_path", type=str, help="Comma-separated list of directories containing kern files.")
    parser.add_argument("--output_path", type=str, help="The path to a json file to write the tokenized output to.")
    parser.add_argument("--processed_data_dir", type=str, help="Directory containing the processed data. Has tokenized data, token_to_id, and id_to_token")
    args = parser.parse_args()
    
    random.seed(0)
    # print(args)
    output_dir = "/".join(args.output_path.split("/")[:-1])
    output_file = args.output_path.split("/")[-1]
    print("input_path: ", args.input_path, "output_path: ", args.output_path, "processed_data_dir: ", args.processed_data_dir)
    
    assert output_file.split(".")[-1] == "json" # ensure that the output path is a json file
        
    assert args.input_path.split(".")[-1] == "xml" # ensure that the input path is a musicxml file
    
    token_to_id, _, metadata = open_processed_data_dir(args.processed_data_dir)
    # print(token_to_id)
    for key, value in token_to_id.items():
        token_to_id[key] = int(value) if value not in CONST_TOKENS else value
    
    parts = converter.parse(args.input_path).parts

    rhythms_and_expressions = {}
    total_num_parts = len(parts)
    num_useful_parts = 0
    for i, part in enumerate(parts):
        tmp = get_rhythms_and_expressions(part=part, want_barlines=metadata["want_barlines"], no_expressions=metadata["no_expressions"], debug=True)

        if tmp is not None:
            rhythms_and_expressions[i+1] = tmp
            num_useful_parts += 1
        else:
            print(f"Skipping part {i+1}")
    print(f"Number of useful parts: {num_useful_parts}/{total_num_parts}")
        
    # now actually tokenize the rhythms and expressions
    num_train_unks = 0
    num_train_tokens = 0
    
    rhythms_and_expressions_tokenized = {}
    for part, notes in rhythms_and_expressions.items():
        rhythms_and_expressions_tokenized[part] = []            
        for note in notes:
            if note in token_to_id.keys():
                rhythms_and_expressions_tokenized[part].append(token_to_id[note])
            else:
                rhythms_and_expressions_tokenized[part].append(token_to_id[UNKNOWN_TOKEN])
    
    # save the tokenized output and dictionaries as pickle files
    print("Writing tokenized output to file...")
    
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w") as f:
        f.write(serialize_json(rhythms_and_expressions_tokenized))