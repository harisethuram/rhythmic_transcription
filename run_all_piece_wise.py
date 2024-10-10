import subprocess
import pandas as pd
import os

if __name__ == "__main__":
    data = pd.read_csv("metadata/URMP/metadata.csv")
    
    for i, row in data.iterrows():
        if row["split"] == "val":
            subprocess.call(["bash", "run/piece_wise_linear_fit.sh", str(row["notes_path"]), str(row["score_path"]), str(row["part_id"]), f"results/URMP/{row['piece_id']}_{row['part_id']}"])