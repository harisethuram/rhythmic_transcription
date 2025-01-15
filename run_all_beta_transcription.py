import subprocess
import pandas as pd
import os
import json

if __name__ == "__main__":
    data = pd.read_csv("metadata/URMP/metadata.csv")
    # transcription_dir = "results/URMP"
    errors = {}
    root_result_dir = "test_results/URMP/"
    base_values = [1/16 , 1/8, 1/4, 1/2, 1]
    test_lim = 10000
    
    for i, row in data.iterrows():
        if i > test_lim: 
            break
        if row["split"] == "val":
            curr_errors = {}
            for base_value in base_values:
                curr_dir = f"{row['piece_id']}_{row['piece_name']}_{row['part_id']}"
                input_path = os.path.join(root_result_dir, curr_dir, "note_info.json")
                output_dir = os.path.join(root_result_dir, curr_dir, f"base_value_{str(base_value)}")
                score_path = str(row["score_path"])
                part_id = str(row["part_id"])
                subprocess.call(["bash", "run/beta_transcription.sh", input_path, output_dir, score_path, part_id, str(base_value)])
                # errors[curr_dir] = float(json.load(open(os.path.join(output_dir, "results.json"), "r"))["error"])
                curr_errors[str(base_value)] = float(json.load(open(os.path.join(output_dir, "results.json"), "r"))["note_error"])
            errors[curr_dir] = curr_errors
    # print(errors)    
    # errors["total"] = sum(errors.values())/len(errors.values())
    json.dump(errors, open(os.path.join(root_result_dir, "beta_total.json"), "w"), indent=4)