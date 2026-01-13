import json
import ast
from music21 import stream, note as m21note, tempo as m21tempo
from src.note.Note import Note
import pandas as pd
from tqdm import tqdm
# If you already have a Note class with .get_len(), import it instead:
# from your_module import Note


def parse_event(event_str):
    """
    event_str is like:
      "('Note(duration=1/2, dotted=True, ... is_rest=False)', 'C#4')"
    We first literal_eval to get:
      ("Note(duration=1/2, dotted=True, ... is_rest=False)", "C#4")
    Then eval the Note(...) part to get your Note object.
    """
    note_repr, pitch = ast.literal_eval(event_str)  # tuple of (str, str or None)
    # note_repr is e.g. "Note(duration=1/2, dotted=True, ...)"

    # This assumes you have a Note class in scope whose constructor matches this string.
    n_obj = eval(note_repr)  # rely on your own Note class

    return n_obj, pitch


def sequence_to_stream(seq, tempo_bpm=120.0):
    s = stream.Stream()

    # Set tempo
    mm = m21tempo.MetronomeMark(number=tempo_bpm)
    s.append(mm)

    for ev in seq:
        # Skip special tokens
        if ev.startswith("('<SOS>'") or ev.startswith("('<EOS>'"):
            continue

        n_obj, pitch = parse_event(ev)

        ql = n_obj.get_len()  # should return music21 quarterLength

        if getattr(n_obj, "is_rest", False) or pitch is None:
            m = m21note.Rest(quarterLength=ql)
        else:
            m = m21note.Note(pitch, quarterLength=ql)

        # Very simple: ignore ties, triplets, articulations, etc.
        s.append(m)

    return s


def main():
    # Replace 'input.json' with your actual file path
    metadata = pd.read_csv("metadata/URMP/metadata.csv")
    metadata = metadata[metadata["split"] == "val"]
    lambdas = ["0.1", "0.3", "0.5", "0.7", "1.0"]
    sigmas = ["0.05", "0.2", "0.5"]
    base_path = "presentation_data/transcriptions/jsons/beam_width_10/lambda_{lbda}/sigma_{sigma}/{piece_id}_{piece_name}_{part_id}/decoded_sequence.json"
    output_path = base_path.replace("decoded_sequence.json", "decoded_sequence.mid")
    for lbda in lambdas:
        for sigma in sigmas:
            for i, row in tqdm(metadata.iterrows(), total=len(metadata)):
                with open(base_path.format(lbda=lbda, sigma=sigma, piece_id=row["piece_id"], piece_name=row["piece_name"], part_id=row["part_id"]), "r") as f:
                    data = json.load(f)

                seq = data["sequence"]
                tempo_bpm = data.get("tempo", 120.0)

                s = sequence_to_stream(seq, tempo_bpm=tempo_bpm)

                # Write to MIDI
                s.write("midi", fp=output_path.format(lbda=lbda, sigma=sigma, piece_id=row["piece_id"], piece_name=row["piece_name"], part_id=row["part_id"]))
                # print("Wrote MIDI")


if __name__ == "__main__":
    main()