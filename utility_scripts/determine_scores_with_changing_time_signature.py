# determine scores with changing time signature
import os
import sys
from music21 import converter, stream, meter
import pandas as pandas

def get_measure_lengths(score_path):
    """
    Get the lengths of measures from a score.
    :param score_path: Path to the score file.
    :return: List of measure lengths.
    """
    try:
        score = converter.parse(score_path).parts[0]
        
        measure_lengths = set()
        
        for meas in score.getElementsByClass('Measure'):
            if meas.barDuration is not None:
                measure_lengths.add(meas.barDuration.quarterLength)
            else:
                # Extremely rare: no barDuration (e.g., TS suppressed).
                # Fall back to the previous time signature context.
                ts = meas.getContextByClass('TimeSignature')
                measure_lengths.add(ts.barDuration.quarterLength if ts else None)

        return measure_lengths

    except Exception as e:
        print(f"Error processing {score_path}: {e}")
        return None

if __name__ == "__main__":
    metadata = pandas.read_csv("metadata/URMP/metadata.csv")
    metadata = metadata[metadata["split"] == "val"]
    valids = []
    for index, row in metadata.iterrows():
        score_path = row["score_path"].split(".")[0] + ".mid"
        measure_lengths = get_measure_lengths(score_path)
        if measure_lengths is not None:
            print(f"Measure lengths for {score_path}: {measure_lengths}")
            valids.append(len(measure_lengths) == 1)
        else:
            print(f"Failed to get measure lengths for {score_path}")    
            valids.append(False)
            
        # add valids as a new column to the metadata
    print(sum(valids), "out of", len(valids), "valid scores")
    metadata["valids"] = valids
    new_path = os.path.join("metadata", "URMP", "new", "metadata.csv")
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    metadata.to_csv(new_path, index=False)
