import pandas as pd
import os
from tqdm import tqdm

if __name__ == "__main__":
    metadata = pd.read_csv("metadata/URMP/metadata.csv")
    new_root_dir = "presentation_data/val_performances"
    metadata = metadata[metadata["split"] == "val"]
    
    os.makedirs(new_root_dir, exist_ok=True)
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
        src_path = row["notes_path"].replace(".txt", ".midi")
        dst_path = os.path.join(new_root_dir, os.path.basename(src_path))
        dist_path = os.path.join(new_root_dir, f"Perf_{row['piece_id']}_{row['piece_name']}_{row['part_id']}.midi")
        os.system(f"cp '{src_path}' '{dist_path}'")
    print("Copied validation performances to", new_root_dir)