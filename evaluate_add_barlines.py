# evaluate barline adding algorithm on ground truth samples by comparing all tokes up through the second barline TODO
import os
import json
import argparse

from src.utils import open_processed_data_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_path", type=str, required=True)
    parser.add_argument("--ground_truth_path", type=str, required=True)
    parser.add_argument("--processed_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    # Load processed data
    processed_data = open_processed_data_dir(args.processed_data_dir)

    # Load predictions
    with open(args.prediction_path, "r") as f:
        predictions = json.load(f)
        
    