# get metadata of schema [piece_id, piece_name, part_id, instrument, notes_path, score_path, split],  primary key: [piece_id, part_id]

import pandas as pandas
import os
import random
import numpy as np

if __name__ == "__main__":
    data_dir = "data/URMP"
    supp = "Supplementary_Files"
    dirs = [f.name for f in os.scandir(data_dir) if f.is_dir() and f.name != supp]
    random.seed(0)
    
    val_len = int(0.5 * len(dirs))
    test_len = len(dirs) - val_len
    metadata = []
    
    for d in dirs:
        piece_id = int(d.split("_")[0])        
        piece_name = d.split("_")[1]
        score_name = os.path.join(data_dir, d, f"Sco_{d}.mid")
        for file in os.listdir(os.path.join(data_dir, d)):
            if file.startswith("Notes"):
                part_id = int(file.split("_")[1])
                instrument = file.split("_")[2]
                metadata.append([piece_id, piece_name, part_id, instrument, os.path.join(data_dir, d, file), score_name])
            
    
    df = pandas.DataFrame(metadata, columns=["piece_id", "piece_name", "part_id", "instrument", "notes_path", "score_path"])
    
    # add split column, where rows of same piece_name are in the same split
    piece_name_to_split = {piece_name: random.choice(["val", "test"]) for piece_name in df["piece_name"].unique()}
    
    splits = [piece_name_to_split[piece_name] for piece_name in df["piece_name"]]
    df.insert(6, "split", splits)    
    
    # df.insert(6, "split", split)
    df = df.sort_values(["piece_id", "part_id"])
    df.to_csv("metadata/URMP/metadata.csv", index=False)
    
    print(len(df[df["split"] == "val"]), len(df[df["split"] == "test"]))
    print(df)
