import pandas as pd
import os

if __name__ == "__main__":
    m_path = "metadata/URMP/metadata.csv"
    metadata = pd.read_csv(m_path)
    
    new_notes_path = []
    new_score_path = []
    for _, row in metadata.iterrows():
        root_dir = "/".join(row["notes_path"].split("/")[:-2])
        notes_file = "/".join(row["notes_path"].split("/")[-2:])
        score_file = "/".join(row["score_path"].split("/")[-2:])
        new_notes_path.append(os.path.join(root_dir, row["split"], notes_file))
        new_score_path.append(os.path.join(root_dir, row["split"], score_file))
        # print(new_notes_path[-1])
        # print(new_score_path[-1])
        # input()
    metadata["notes_path"] = pd.DataFrame(new_notes_path)
    metadata["score_path"] = pd.DataFrame(new_score_path)
    metadata.to_csv(m_path, index=False)
    