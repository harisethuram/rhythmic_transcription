import subprocess
import pandas as pd
import os
import json

if __name__ == "__main__":
    data = pd.read_csv("metadata/URMP/metadata.csv")
    # transcription_dir = "results/URMP"
    errors = {}
    root_result_dir = "output/test_results/URMP/"
    base_values = [1/2, 1, 2, 4]
    methods = ["beam_search", "greedy"]
    test_lim = 10000
    
    for i, row in data.iterrows():
        if i > test_lim: 
            break
        if row["split"] == "val":
            curr_errors = {}
            for base_value in base_values:
                for method in methods:
                    print(f"Running {base_value} {method} on {row['piece_id']}_{row['piece_name']}_{row['part_id']}")
                    curr_dir = f"{row['piece_id']}_{row['piece_name']}_{row['part_id']}"
                    input_path = os.path.join(root_result_dir, curr_dir, "note_info.json")
                    output_dir = os.path.join(root_result_dir, curr_dir, f"decoder/base_value_{str(base_value)}/{method}")
                    score_path = str(row["score_path"])
                    part_id = str(row["part_id"])
                    subprocess.call(["bash", "run/decode_custom.sh", input_path, output_dir, str(base_value), method])
                    subprocess.call(["bash", "run/to_mxl.sh", os.path.join(output_dir, "output.json"), input_path, output_dir])
                    
                    break
                break