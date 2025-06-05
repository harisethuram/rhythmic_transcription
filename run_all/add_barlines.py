import pandas as pd
import subprocess
from tqdm import tqdm
import os

if __name__ == "__main__":
    metadata_path = "metadata/URMP/metadata.csv"
    data = pd.read_csv(metadata_path)
    data = data[data["split"] == "val"]
    base_note_info_path = "output/presentation_results/ablations/perfect_piecewise/piecewise"
    base_input_path = "output/presentation_results/ablations/perfect_decode/decode"
    base_output_path = "output/presentation_results/ablations/perfect_decode/added_barlines"
    
    for i, row in tqdm(data.iterrows()):
        # for base_value in ["0.25", "0.5", "1", "2", "3", "4"]:
        print(f"Processing {row['piece_id']}_{row['piece_name']}_{row['part_id']} with base value {1}")
        piece = f"{row['piece_id']}_{row['piece_name']}_{row['part_id']}"
        # input_path = os.path.join("output", "test_decode", "URMP", "decodes", piece, f"base_value_{base_value}", "output.json")
        input_path = os.path.join(base_input_path, f"{row['piece_id']}_{row['piece_name']}", f"part_{row['part_id']}.json")
        output_path = os.path.join(base_output_path, f"{piece}.json")
        note_info_path = os.path.join(base_note_info_path, f"{piece}.json")

        if os.path.exists(output_path):
            print(f"Output path already exists: {output_path}")
            continue
        
        subprocess.call(["bash", "run/add_barlines.sh", input_path, output_path, note_info_path])