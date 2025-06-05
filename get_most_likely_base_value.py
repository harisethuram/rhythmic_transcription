import pandas as pd
import json
import os

if __name__ == "__main__":
    metadata = pd.read_csv("metadata/URMP/metadata.csv")
    metadata = metadata[metadata["split"] == "val"]
    results = {}
    base_values = ["0.25", "0.5", "1", "2", "3", "4"]
    base_path = "output/test_decode/URMP/added_barlines"
    for i, row in metadata.iterrows():
        piece = f"{row['piece_id']}_{row['piece_name']}_{row['part_id']}"
        results[piece] = []
        for base_value in base_values:  
            this_path = os.path.join(base_path, piece + "_" + base_value + ".json")
            if os.path.exists(this_path):
                with open(this_path, "r") as f:
                    data = json.load(f)
                    results[piece].append((base_value, float(data["log_likelihood"])))

        results[piece] = sorted(results[piece], key=lambda x: x[1], reverse=True)
    
    with open("basevalues.json", "w") as f:
        json.dump(results, f, indent=4)
                