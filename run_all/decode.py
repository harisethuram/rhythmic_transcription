import subprocess
import pandas as pd
import os
import json

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all decoding")
    parser.add_argument("chunk", type=int, help="Chunk number to run")
    args = parser.parse_args()
    data = pd.read_csv("metadata/URMP/metadata.csv")
    
    errors = {}
    root_result_dir = "output/presentation_results/ablations/perfect_piecewise/decode"
    base_values = [1]
    # base_values = [2, 4]
    methods = ["beam_search"]
    test_lim = 10000
    base_note_info_dir = "output/presentation_results/ablations/perfect_piecewise/piecewise"
    processed_data_dir = "processed_data/all/no_barlines"
    all_results = {}
    default_high = 10000
    
    data = data[data["split"] == "val"]
    
    full_length = len(data)
    num_chunks = 8
    chunk_size = len(data) // num_chunks
    chunks = [data.iloc[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    data = chunks[args.chunk]
    
    # split into chunks
    # print(len(data))
    # data = data.iloc[chunk_size * args.chunk : chunk_size * (args.chunk + 1)]
    print("length: {} / {}".format(len(data), full_length))
    for i, row in data.iterrows():
        print("row {}".format(i))
        # if i > test_lim: 
        #     break
        if row["split"] == "val":
            print(f"Running {row['piece_id']}_{row['piece_name']}_{row['part_id']}")
            curr_errors = {}
            best_base_value = -1
            best_accuracy = -1
            best_length_se = default_high   
            for base_value in base_values:
                # for method in methods:
                
                print(f"Running {base_value} beam_search on {row['piece_id']}_{row['piece_name']}_{row['part_id']}")
                curr_dir = f"{row['piece_id']}_{row['piece_name']}_{row['part_id']}"
                note_info_path = os.path.join(base_note_info_dir, curr_dir + ".json")
                output_dir = os.path.join(root_result_dir, curr_dir)
                # score_path = os.path.join(processed_data_dir, f"{row['piece_id']}_{row['piece_name']}.json")
                # part_id = str(row["part_id"])
                # if os.path.exists(os.path.join(output_dir, "results.json")):
                #     print("skipping")
                #     continue
                # alignment_path = os.path.join(base_note_info_dir, curr_dir, "alignment.json")
                subprocess.call(["bash", "run/decode.sh", note_info_path, output_dir, str(base_value)])
                    
                # get best base value
                # results_path = os.path.join(output_dir, "results.json")
                # if os.path.exists(results_path):
                #     results = json.load(open(results_path, "r"))
                #     if float(results["total_binary_accuracy"]) > best_accuracy:
                #         best_accuracy = float(results["total_binary_accuracy"])
                #         best_base_value = base_value
                #     if float(results["total_length_se"]) < best_length_se:
                #         best_length_se = float(results["total_length_se"])
                        
                # all_results[curr_dir] = {
                #     "best_base_value": best_base_value,
                #     "best_accuracy": best_accuracy,
                #     "best_length_se": best_length_se
                # }
        # break
    # don't include -1s or infs in the average
    # all_results["total"] = {
    #     "avg_best_accuracy": str(sum([m["best_accuracy"] for m in all_results.values() if m["best_accuracy"] != -1])/len([m["best_accuracy"] for m in all_results.values() if m["best_accuracy"] != -1])),
    #     "avg_best_length_se": str(sum([m["best_length_se"] for m in all_results.values() if m["best_length_se"] != default_high])/len([m["best_length_se"] for m in all_results.values() if m["best_length_se"] != default_high]))
    # }    
    # for song, result in all_results.items():
    #     all_results[song]["best_base_value"] = str(result["best_base_value"])
    #     all_results[song]["best_accuracy"] = str(result["best_accuracy"])
    #     all_results[song]["best_length_se"] = str(result["best_length_se"])
        
            
    # print("TOTAL:")
    # print(all_results["total"])
    # # save
    # json.dump(all_results, open(os.path.join(root_result_dir, "total.json"), "w"), indent=4)
                #     break
                # break