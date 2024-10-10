import json
import argparse
import pandas as pd
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the tokenized dataset")
    parser.add_argument("--output_path", type=str, help="Path to save the metadata")
    
    args = parser.parse_args()
    
    with open(args.data_path, "r") as f:
        data = json.load(f)
        
    metadata = []
    for piece_name, items in data.items():
        
        for part in items.keys():
            if part == "split":
                continue
            metadata.append({"piece_name": piece_name, "part_id": part, "num_tokens": len(items[part]), "split": items["split"], "path": os.path.join("data", "bach-370-chorales", "kern", piece_name+".krn")})
                
    metadata = pd.DataFrame(metadata).sort_values(by=["split", "piece_name", "part_id"])
    metadata.to_csv(args.output_path, index=False)
    
    