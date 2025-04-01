import subprocess
import os
import pandas as pd
from tqdm import tqdm
import json

if __name__ == "__main__":
    metadata = pd.read_csv("metadata/URMP/metadata.csv")
    
    split="val"
    root_perf_midi_dir = f"musescore/musescore_perf_midi/{split}"
    root_transcribe_midi_dir = f"musescore/transcribe_midi/{split}"
    root_transcribe_xml_dir = f"musescore/transcribe_xml/{split}"
    
    root_alignment_dir = f"musescore/alignments/{split}"
    root_tokens_dir = f"musescore/tokens/{split}"
    root_output_dir = f"musescore/eval/{split}"
    
    root_gt_score_json_dir = "processed_data/URMP_test_all/validation"
    
    total = {}
    num_valid = 0
    
    metrics = ["total_binary_accuracy", "total_length_se", "total_correct_given_length_se_0"]
    test_lim = 1000
    count = -1
    for i, row in tqdm(metadata.iterrows()):
        # count += 1
        # if count == 0 or count == 1:
        #     continue
        if i >= test_lim:
            break
        if row["split"] != "val":
            continue
        piece_name = row["notes_path"].split(".")[0].split("/")[-1]
        print(piece_name)
        # first, get the alignment
        prediction_midi_path = os.path.join(root_transcribe_midi_dir, piece_name + ".mid")
        gt_midi_path = row["score_path"].split(".")[0] + ".mid"
        part_id = row["part_id"]
        alignment_output_dir = os.path.join(root_alignment_dir, piece_name)
        alignment_output_path = os.path.join(alignment_output_dir, "alignment.json")
        subprocess.call(["bash", "run/generate_alignment_given_midi.sh", prediction_midi_path, gt_midi_path, str(part_id), alignment_output_dir])
        
        # second, tokenize the xml files
        input_path = os.path.join(root_transcribe_xml_dir, piece_name+".xml")
        output_path = os.path.join(root_tokens_dir, piece_name+".json")
        subprocess.call(["bash", "run/mxl_tokenizer.sh", input_path, output_path])
        
        # third, do the actual evaluation
        score_path = os.path.join(root_gt_score_json_dir, f"{row["piece_id"]}_{row["piece_name"]}.json")
        # with open(score_path, "r") as f:
        #     score_json_tmp = json.load(score_path)
        #     # if len(score_json_tmp[str(score_part_id)]) > 600
        score_part_id = row["part_id"]
        transcription_path = output_path
        transcription_part_id = 1
        alignment_path = alignment_output_path
        output_dir = os.path.join(root_output_dir, piece_name)
        processed_data_dir = "processed_data/test_all"
        subprocess.call(["bash", "run/decode_eval.sh", score_path, str(score_part_id), transcription_path, str(transcription_part_id), alignment_path, output_dir, processed_data_dir])
        
        # compile
        curr_results = {metric: -1 for metric in metrics}
        output_path = os.path.join(output_dir, "results.json")
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                tmp = json.load(f)
                curr_results = {metric: float(tmp[metric]) for metric in metrics}
                num_valid += 1
        else:
            print("ERROR!!")
        total[piece_name] = curr_results
        # input()
    total["average"] = {metric: sum([val[metric] for val in total.values() if val[metric] != -1]) / num_valid for metric in metrics}
    total_output_path = os.path.join(root_output_dir, "total.json")
    
    with open(total_output_path, "w") as f:
        json.dump(total, f)