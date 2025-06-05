# converts an xml file to a tokenized json file

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
from src.utils import serialize_json, open_processed_data_dir, decompose_note_sequence_notes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tokenizes a musicxml file")
    parser.add_argument("--input_path", type=str, help="Path to xml file.")
    parser.add_argument("--output_dir", type=str, help="The path to a json file to write the tokenized output to.")
    parser.add_argument("--processed_data_dir", type=str, help="Directory containing the processed data. Has tokenized data, token_to_id, and id_to_token")
    args = parser.parse_args()
    
    print(args)
    random.seed(0)
    # print(args)
    # output_dir = "/".join(args.output_path.split("/")[:-1])
    # output_file = args.output_path.split("/")[-1]
    # print("input_path: ", args.input_path, "output_path: ", args.output_path, "processed_data_dir: ", args.processed_data_dir)
    
    # assert output_file.split(".")[-1] == "json" # ensure that the output path is a json file
        
    # assert args.input_path.split(".")[-1] == "xml" # ensure that the input path is a musicxml file
    
    token_to_id, id_to_token, metadata = open_processed_data_dir(args.processed_data_dir)

    for key, value in token_to_id.items():
        token_to_id[key] = int(value) if value not in CONST_TOKENS else value
    
    parts = converter.parse(args.input_path).parts

    rhythms_and_expressions = {}
    total_num_parts = len(parts)
    num_useful_parts = 0
    want_barlines = metadata["want_barlines"] == "True"
    no_expressions = metadata["no_expressions"] == "True"
    for i, part in enumerate(parts):
        tmp = get_rhythms_and_expressions(part=part, want_barlines=False, no_expressions=bool(metadata["no_expressions"]), debug=True)[0]
        # print("timp: ", tmp)
        # input()
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
        notes = notes[0]
        rhythms_and_expressions_tokenized[part] = {"notes": [], "tokens": []}   
        # print("notes: ", notes)      
        # input()   
        for note in notes:
            
            # print(note)
            # note = note[0]
            if note in token_to_id.keys():
                rhythms_and_expressions_tokenized[part]["notes"].append(note)
                rhythms_and_expressions_tokenized[part]["tokens"].append(token_to_id[note])
            else:
                rhythms_and_expressions_tokenized[part]["notes"].append(note)
                rhythms_and_expressions_tokenized[part]["tokens"].append(token_to_id[UNKNOWN_TOKEN])

    # save the tokenized output and dictionaries as json files
    print("Writing tokenized output to file...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        f.write(serialize_json(rhythms_and_expressions_tokenized))
    
    for part in rhythms_and_expressions_tokenized.keys():
        with open(os.path.join(args.output_dir, f"part_{part}.json"), "w") as f:
            f.write(serialize_json(decompose_note_sequence_notes(rhythms_and_expressions_tokenized[part]["notes"], token_to_id, id_to_token)))
    