import pandas as pd

def gen_path(row):
    path = row["notes_path"]
    p = "/".join(path.split("/")[:-1])
    p += f"/Align_{row['part_id']}_{row['piece_name']}_{row['piece_id']}.json"
    return p

if __name__ == "__main__":
    metadata_p = "metadata/URMP/metadata.csv"
    metadata = pd.read_csv(metadata_p)
    alignment_paths = [gen_path(row) for _, row in metadata.iterrows()]
    print(alignment_paths)
    metadata["alignment_path"] = alignment_paths
    metadata.to_csv(metadata_p, index=False)