import json
import os
import argparse
from src.utils import serialize_json
from const_tokens import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create parallel barline dataset")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the barline dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output dataset")
    
    args = parser.parse_args()
    
    data = json.load(open(os.path.join(args.input_dir, "tokenized_dataset.json"), "r"))
    id_to_token = json.load(open(os.path.join(args.input_dir, "id_to_token.json"), "r"))
    token_to_id = json.load(open(os.path.join(args.input_dir, "token_to_id.json"), "r"))
    
    parallel_data = {} 
    
    for piece_path, piece in data.items():
        parallel_data[piece_path] = {}
        for part_id, part in piece.items():
            if part_id == "split":
                continue
            
            parallel_data[piece_path][part_id] = {
                "barline": part,
                "no_barline": [token for token in part if token != token_to_id[BARLINE_TOKEN]],
            }
        parallel_data[piece_path]["split"] = piece["split"]
        
    # Save the new dataset
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, "parallel_barline_dataset.json"), "w") as f:
        f.write(serialize_json(parallel_data))
    with open(os.path.join(args.output_dir, "token_to_id.json"), "w") as f:
        f.write(serialize_json(token_to_id))
    with open(os.path.join(args.output_dir, "id_to_token.json"), "w") as f:
        f.write(serialize_json(id_to_token))
    # also save the tokenizers for with barline
    