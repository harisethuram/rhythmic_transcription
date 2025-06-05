import pandas as pd
import os
import json
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess.ingestion_utils import get_note_info_from_xml
from src.utils import serialize_json


if __name__ == "__main__":
    metadata = pd.read_csv("metadata/URMP/metadata.csv")
    metadata = metadata[metadata["split"] == "val"]
    for index, row in tqdm(metadata.iterrows()):
        xml_file_path = row["score_path"]
        part_id = row["part_id"]
        note_info = get_note_info_from_xml(xml_file_path, part_id)
        
        with open(os.path.join("output/presentation_results/ablations/perfect_piecewise", f"{row['piece_id']}_{row['piece_name']}_{row['part_id']}.json"), "w") as f:
            f.write(serialize_json(note_info))
        # print(row['piece_name'], part_id)
        # input()