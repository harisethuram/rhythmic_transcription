# tokenizes the URMP val split to see how well the pretraining dataset covers the URMP val split.
import os
import argparse
import pandas as pd
import sys
import numpy as np
import pickle as pkl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess.ingestion_utils import get_score_note_lengths
from src.preprocess.tokenizer_utils import analyze_duration

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--processed_data_dir", type=str, , help="The directory containing the processed data.")
    # args = parser.parse_args()
    
    processed_data_dir = "processed_data/bach_fugues"
    
    metadata = pd.read_csv("metadata/URMP/metadata.csv")
    val_split = metadata[metadata["split"] == "val"]
    
    scores = list(set(val_split["score_path"].values))
    scores.sort()
    print(scores)
    
    token_to_id = pkl.load(open(os.path.join(processed_data_dir, "token_to_id.pkl"), "rb"))
    
    for score in scores:
        print(score)
        score_notes, is_note = get_score_note_lengths(score)
        print(score_notes)
        print(is_note)
        input()

        # for onset_lengths in onset_lengths_all_parts:
        #     for onset_length in onset_lengths:
        #         base_value, dot_value, is_dotted, is_triplet = analyze_duration(onset_length)
        #         token = get_tuple(base)
        
        
    
    
    