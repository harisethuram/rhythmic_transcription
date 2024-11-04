import subprocess
import pandas as pd
import os
import json

if __name__ == "__main__":
    data = pd.read_csv("metadata/URMP/metadata.csv")
    errors = {}
    root_result_dir = "results/URMP/"
    for i, row in data.iterrows():
        if row["split"] == "val":
            curr_dir = f"{row['piece_id']}_{row['piece_name']}_{row['part_id']}"
            output_dir = os.path.join(root_result_dir, curr_dir)
            subprocess.call(["bash", "run/piece_wise_linear_fit.sh", str(row["notes_path"]), str(row["score_path"]), str(row["part_id"]), output_dir])
            errors[curr_dir] = float(json.load(open(os.path.join(output_dir, "results.json"), "r"))["error"])
            
    errors["total"] = sum(errors.values())/len(errors.values())
    json.dump(errors, open(os.path.join(root_result_dir, "total.json"), "w"), indent=4)