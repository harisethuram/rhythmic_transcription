import json

import os

if __name__ == "__main__":
    files = os.listdir("/mmfs1/gscratch/ark/hari/rhythmic_trascription/presentation_data/musescore/alignments")
    for f in files:
        if f.endswith(".json"):
            path = os.path.join("/mmfs1/gscratch/ark/hari/rhythmic_trascription/presentation_data/musescore/alignments", f)
            data = json.load(open(path, "r"))
            data = [(int(score_id), int(perf_id)) for score_id, perf_id in data] 
            txt_path = path.replace(".json", ".txt")
            with open(txt_path, "w") as fout:
                for score_id, perf_id in data:
                    fout.write(f"{score_id},{perf_id}\n")