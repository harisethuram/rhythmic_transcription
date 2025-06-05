import pandas as pd
import os
import json
import ast
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import serialize_json

if __name__ == "__main__":
    data = pd.read_csv("metadata/URMP/metadata.csv")
    data = data[data["split"] == "val"]
    errors = {}
    # root_result_dir = "output/presentation_results/ablations/perfect_decode"
    base_dir = "output/presentation_results/ablations/perfect_piecewise/piecewise"
    for i, row in data.iterrows():
        curr_path = os.path.join(base_dir, f"{row['piece_id']}_{row['piece_name']}_{row['part_id']}.json")
        with open(curr_path, "r") as f:
            data = json.load(f)
            
        data = [ast.literal_eval(x) for x in data]
        # print(data)
        # input()
        with open(curr_path, "w") as f:
            json.dump(data, f)