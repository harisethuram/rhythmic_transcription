import subprocess
import pandas as pd
import os
import json
from tqdm import tqdm

if __name__ == "__main__":
    metadata_path = "metadata/URMP/metadata.csv"
    data = pd.read_csv(metadata_path)
    perf_midi_paths = []
    errors = {}
    for i, row in tqdm(data.iterrows()):
        output_path = os.path.join("output", "musescore_perf_midi", row["split"],  row["notes_path"].split(".")[0].split("/")[-1] + ".midi")
        perf_midi_paths.append(output_path)
        subprocess.call(["bash", "run/convert_txt_to_midi.sh", row["notes_path"], output_path])

    # print(len(data), len(perf_midi_paths), perf_midi_paths)
    # data["perf_midi_paths"] = perf_midi_paths
    # data.to_csv("test/test.csv", index=False)
    
    