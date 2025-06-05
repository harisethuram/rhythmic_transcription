import json
import os
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.note.Note import Note
from src.utils import open_processed_data_dir, serialize_json

if __name__ == "__main__":
    metadata = pd.read_csv("metadata/URMP/metadata.csv")
    
    token_to_id, id_to_token, _ = open_processed_data_dir("processed_data/all/barlines")
    
    for i, row in metadata.iterrows():
        for base_value in ["0.25", "0.5", "1", "2", "3", "4"]:
            curr_path = os.path.join("output/test_decode/URMP/added_barlines", f"{row['piece_id']}_{row['piece_name']}_{row['part_id']}_{base_value}.json")
            if os.path.exists(curr_path):
                with open(curr_path, "r") as f:
                    data = json.load(f)
                    
                    data["sequence"] = [id_to_token[int(id)] for id in data["sequence"][:-1]]
                    data["pitches"] = data["pitches"][1:-1]
                    
                with open(curr_path, "w") as f:
                    f.write(serialize_json(data))