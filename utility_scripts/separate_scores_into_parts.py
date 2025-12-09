import pandas as pd
import os
import music21 as m21
from tqdm import tqdm

def split_score_into_parts(score_path, output_path, wanted_part_id):
    """
    Load a multi-part MusicXML score and save exactly one selected part.

    Parameters
    ----------
    score_path : str
        Path to the input .xml/.musicxml file.

    output_path : str
        Path where the extracted single-part MusicXML file will be written.

    wanted_part_id : int
        1-indexed integer selecting which part to extract.
        Example: 1 = first part, 2 = second part, etc.
    """

    # Parse the score
    score = m21.converter.parse(score_path)

    # Safety: ensure valid index
    num_parts = len(score.parts)
    if wanted_part_id < 1 or wanted_part_id > num_parts:
        raise ValueError(
            f"wanted_part_id={wanted_part_id} is out of range. "
            f"Score contains {num_parts} parts."
        )

    # Select part (convert 1-indexed to 0-indexed)
    part = score.parts[wanted_part_id - 1]

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write out the selected part
    part.write("musicxml", fp=output_path)

    print(f"Saved part {wanted_part_id} → {output_path}")


if __name__ == "__main__":
    metadata = pd.read_csv("metadata/URMP/metadata.csv")
    
    # iterate over rows
    new_col = []
    for index, row in tqdm(metadata.iterrows(), total=len(metadata)):
        score_path = row["score_path"]
        
        output_path = os.path.join(*score_path.split("/")[:-1]) + f"/Sco_{row['piece_id']}_{row['piece_name']}_" + f"{row['part_id']}.xml"
        try:
            if not os.path.exists(output_path):
                split_score_into_parts(score_path, output_path, int(row['part_id']))
            
        except Exception as e:
            print(f"Error processing {score_path} part {row['part_id']}: {e}")
        new_col.append(output_path)
        # print(output_path_format)
        # input()
    
    metadata["score_parts"] = new_col
    metadata.drop(columns=["alignment_path"], inplace=True)
    metadata.to_csv("metadata/URMP/metadata_with_parts.csv", index=False)