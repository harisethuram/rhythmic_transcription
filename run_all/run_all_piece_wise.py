import subprocess
import pandas as pd
import os
import json
from statistics import median

if __name__ == "__main__":
    data = pd.read_csv("metadata/URMP/metadata.csv")
    errors = {}
    root_result_dir = "debug_results/"
    
    test_lim = 10000
    for i, row in data.iterrows():
        if i > test_lim:
            break
        if row["split"] == "val":
            curr_dir = f"{row['piece_id']}_{row['piece_name']}_{row['part_id']}"
            output_dir = os.path.join(root_result_dir, curr_dir)
            subprocess.call(["bash", "run/piece_wise_linear_fit.sh", str(row["notes_path"]), str(row["score_path"]), str(row["part_id"]), output_dir])
            try: 
                results_json = json.load(open(os.path.join(output_dir, "results.json"), "r"))["error"]
                # errors[curr_dir] = {"mean_onset_lengths_diff": results_json["mean_onset_lengths_diff"]}
            except FileNotFoundError:
                print(f"Error in {curr_dir}, not found")
        # break
            
    # errors["total"] = {
    #     "avg_mean_onset_lengths_diff": sum([m["mean_onset_lengths_diff"] for m in errors.values()])/len(errors.values()),
    #     # "avg_mean_onset_times_diff": sum([m["mean_onset_times_diff"] for m in errors.values()])/len(errors.values()),
    #     "median_mean_onset_lengths_diff": median([m["mean_onset_lengths_diff"] for m in errors.values()]),
    #     # "median_mean_onset_times_diff": median([m["mean_onset_times_diff"] for m in errors.values()])
    # }
    # json.dump(errors, open(os.path.join(root_result_dir, "total.json"), "w"), indent=4)