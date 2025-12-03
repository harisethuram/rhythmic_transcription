import pandas as pd
import subprocess
from tqdm import tqdm
import os

if __name__ == "__main__":
    metadata_path = "metadata/URMP/metadata.csv"
    data = pd.read_csv(metadata_path)
    data = data[data["split"] == "val"]
    # base_note_info_path = "output/presentation_results/ablations/perfect_piecewise/piecewise"
    base_note_info_path = "output/presentation_results/piecewise"
    # base_input_path = "output/presentation_results/ablations/perfect_decode/decode"
    base_input_path = "output/presentation_results/test/does_lm_decode_work/mix"
    # base_output_path = "output/presentation_results/ablations/perfect_decode/added_barlines"
    base_output_path = "output/presentation_results/test/does_lm_decode_work/mix_added_barlines_and_measure_lengths"
    model_path = "output/presentation_results/models/barlines_and_measure_lengths/lr_1e-3/b_size_32/emb_64/hid_256/model.pth"
    processed_data_dir = "processed_data/all/barlines_and_measure_lengths"
    
    for i, row in tqdm(data.iterrows()):
        piece = f"{row['piece_id']}_{row['piece_name']}_{row['part_id']}"
        for base_value in ["0.25", "0.5", "1", "2", "3", "4"]:
            print(f"Processing {piece} with base value {base_value}")

            # input_path = os.path.join("output", "test_decode", "URMP", "decodes", piece, f"base_value_{base_value}", "output.json")
            input_path = os.path.join(base_input_path, piece, f"{base_value}", "output.json")
            output_path = os.path.join(base_output_path, piece, f"{base_value}", "output.json")
            note_info_path = os.path.join(base_note_info_path, f"{piece}/note_info.json")

            # if os.path.exists(output_path):
            #     print(f"Output path already exists: {output_path}")
            #     continue
            
            subprocess.call(["bash", "run/add_barlines.sh", input_path, output_path, note_info_path, model_path, processed_data_dir])