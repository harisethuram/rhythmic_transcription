import subprocess
import pandas as pd
import os
import json

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all decoding")
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--processed_data_dir", type=str, help="Path to the processed data directory")
    parser.add_argument("--tempos", type=str, help="Comma separated tempos in BPM")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width for beam search")
    parser.add_argument("--lambda_param", type=float, default=0.5, help="Weight for combining the rhythm LSTM probabilities and Gaussian channel probabilities (1=only gaussian, 0=only LSTM)")
    parser.add_argument("--sigma", type=float, default=0.1, help="Standard deviation for the Gaussian channel")
    parser.add_argument("--root_result_dir", type=str, help="Root directory for results")
    args = parser.parse_args()
    
    metadata = pd.read_csv("metadata/URMP/metadata.csv")
    metadata = metadata[metadata["split"] == "val"]
    
    for i, row in metadata.iterrows():
        input_path = row["notes_path"]
        output_dir = os.path.join(args.root_result_dir, f"{row['piece_id']}_{row['piece_name']}_{row['part_id']}")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Decoding {input_path} to {output_dir}")
        
        subprocess.run(["bash", "run/processes/gaussian_decode.sh", args.model_path, args.processed_data_dir, input_path, args.tempos, output_dir, str(args.beam_width), str(args.lambda_param), str(args.sigma)])
        
        