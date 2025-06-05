# write ground truth for perfected decoding output, so basically remove barlines from the sequence 
import pandas as pd
import os
import json
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess.ingestion_utils import get_note_info_from_xml
from src.utils import serialize_json, decompose_note_sequence, open_processed_data_dir
from src.const_tokens import *

if __name__ == "__main__":
    metadata = pd.read_csv("metadata/URMP/metadata.csv")
    
    metadata = metadata[metadata["split"] == "val"]
    token_to_id, id_to_token, _ = open_processed_data_dir("processed_data/all/barlines")
    
    