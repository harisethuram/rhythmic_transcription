import pandas as pd
import os
from tqdm import tqdm
import subprocess
if __name__ == "__main__":
    metadata = pd.read_csv("metadata/URMP/metadata.csv")
    base_values = ["0.25", "0.5", "1", "2", "3", "4"]
    metadata = metadata[metadata["split"] == "val"]
    for i, row in tqdm(metadata.iterrows()):
        
        piece = f"{row['piece_id']}_{row['piece_name']}_{row['part_id']}"
        for base_value in base_values:
            # input_path = os.path.join("output/presentation_results/ablations/perfect_decode/added_barlines", f"{piece}.json")
            input_path = os.path.join("output/presentation_results/added_barlines_with_measure_lengths", piece, f"base_value_{base_value}", "output.json")
            output_path = os.path.join("output/presentation_results/added_barlines_with_measure_lengths", piece, f"base_value_{base_value}", "output.xml")
            
            if os.path.exists(input_path):
                if os.path.exists(output_path):
                    print(f"XML file already exists for {piece} with base value {base_value}, skipping...")
                    continue
                # print(f"Processing {piece} with base value {base_value}")
                subprocess.call(["python", "tokens_to_xml.py", "--input_path", input_path, "--output_path", output_path])