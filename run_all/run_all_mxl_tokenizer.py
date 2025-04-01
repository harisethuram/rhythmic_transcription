import subprocess
import pandas as pd
import os
import json

if __name__ == "__main__":
    data = pd.read_csv("metadata/URMP/metadata.csv")
    errors = {}
    root_result_dir = "processed_data/URMP_test_all/validation"
    
    test_lim = 10000
    visited = set()
    for i, row in data.iterrows():
        if row["piece_id"] in visited:
            continue
        if i > test_lim:
            break
        if row["split"] == "val":
            curr_path = f"{row['piece_id']}_{row['piece_name']}.json"
            subprocess.call(["bash", "run/mxl_tokenizer.sh", row["score_path"], os.path.join(root_result_dir, curr_path)])
        # input()
        visited.add(row["piece_id"])