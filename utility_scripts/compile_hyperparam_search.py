"""output/gaussian_decoder/hyperparam_search2/beam_width_10/lambda_1.0/sigma_0.2/1_Jupiter_1/decoded_sequence_0.json"""

import pandas as pd
import os
from tqdm import tqdm
import json

# we want to move all decoded_sequence_0.json files to a single folder with informative names. 
# 
if __name__ == "__main__":
    metadata = pd.read_csv("metadata/URMP/metadata.csv")
    metadata = metadata[metadata["split"] == "val"]
    
    lambdas = ["0.1", "0.3", "0.5", "0.7", "1.0"]
    sigmas = ["0.05", "0.2", "0.5"]
    beam_width = "10"
    
    output_root = "presentation_data/transcriptions/jsons"
    os.makedirs(output_root, exist_ok=True)
    
    for lambda_val in lambdas:
        for sigma in sigmas:
            hyperparam_dir = f"output/gaussian_decoder/hyperparam_search2/beam_width_{beam_width}/lambda_{lambda_val}/sigma_{sigma}"
            output_dir = os.path.join(output_root, f"beam_width_{beam_width}/lambda_{lambda_val}/sigma_{sigma}")
            
            for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
                piece_id = row["piece_id"]
                piece_name = row["piece_name"]
                part_id = row["part_id"]
                
                src_path = os.path.join(hyperparam_dir, f"{piece_id}_{piece_name}_{part_id}", "decoded_sequence_0.json")
                dst_path = os.path.join(output_dir, f"{piece_id}_{piece_name}_{part_id}", "decoded_sequence.json")
                
                # dst_filename = f"decoded_sequence_lambda{lambda_val}_sigma{sigma}_{piece_id}_{piece_name}_part{part_id}.json"
                
                # if os.path.exists(src_path) and not os.path.exists(dst_path):
                #     os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                #     os.system(f"cp '{src_path}' '{dst_path}'")
                # elif not os.path.exists(src_path):
                #     print(f"Warning: source json path does not exist: {src_path}")    

                
                # do the same for the pdf files
                pdf_src_path = os.path.join(hyperparam_dir, f"{piece_id}_{piece_name}_{part_id}", "decoded_sequence_0.pdf")
                pdf_dst_path = os.path.join(output_dir, f"{piece_id}_{piece_name}_{part_id}", "decoded_sequence.pdf")
                if os.path.exists(pdf_src_path) and not os.path.exists(pdf_dst_path):
                    os.makedirs(os.path.dirname(pdf_dst_path), exist_ok=True)
                    os.system(f"cp '{pdf_src_path}' '{pdf_dst_path}'")
                elif not os.path.exists(pdf_src_path):
                    print(f"Warning: source pdf path does not exist: {pdf_src_path}")
                # elif os.path.exists(dst_path) and os.path.exists(src_path):
                #     print(f"Destination path already exists: {dst_path}")
                
    