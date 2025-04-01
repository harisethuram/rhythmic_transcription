import pretty_midi
import os
import argparse
import numpy as np
import pandas as pd

def text_to_midi(input_file, output_file):
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    
    # Create an instrument (flute)
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Flute"))
    
    # Read the text file and process each note
    with open(input_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            parts = [part for part in line.strip().split("\t") if len(part) > 0]
            if len(parts) != 3:
                continue  # Skip invalid lines
            
            onset_time = float(parts[0])  # Onset time in seconds
            frequency = float(parts[1])  # Pitch in Hz
            duration = float(parts[2])  # Note duration in seconds
            
            pitch = int(round(pretty_midi.hz_to_note_number(frequency)))
            # Create a note with velocity 100 (range: 0-127)
            note = pretty_midi.Note(velocity=100, pitch=pitch, start=onset_time, end=onset_time + duration)
            
            # Add the note to the instrument
            instrument.notes.append(note)

    # Add the instrument to the PrettyMIDI object
    midi.instruments.append(instrument)

    # Write the MIDI file
    midi.write(output_file)


def test_midi_conversion(text_file, midi_file):
    # Read the original text file into a DataFrame
    txt_file = text_file
    # df = pd.read_csv(txt_file, sep="\t", names=["onset", "pitch", "duration"])
    with open(txt_file, "r") as file:
        lines = file.readlines()
        tmp = []
        for line in lines:
            tmp.append([float(part) for part in line.strip().split("\t") if len(part) > 0])
        tmp = np.array(tmp)
        print("tmp", tmp)
        df = pd.DataFrame({"onset": tmp[:, 0], "pitch": tmp[:, 1], "duration": tmp[:, 2]})
        

    # Convert the text file to MIDI
    output_midi = midi_file
    # txt_to_midi(txt_file, output_midi)  # Ensure this function is defined

    # Load the generated MIDI file
    midi_data = pretty_midi.PrettyMIDI(output_midi)

    # Get the first instrument's notes
    midi_notes = midi_data.instruments[0].notes

    # Ensure the number of notes matches
    assert len(midi_notes) == len(df), f"Expected {len(df)} notes, but found {len(midi_notes)} in MIDI."

    # Compare each note
    for i, note in enumerate(midi_notes):
        expected_onset = int(df.iloc[i]["onset"] * 1000)
        expected_duration = int(df.iloc[i]["duration"] * 1000)
        expected_pitch = int(round(pretty_midi.hz_to_note_number(df.iloc[i]["pitch"])))
        actual_duration = int(note.end * 1000) - int(note.start * 1000)
        print(expected_onset, note.start, expected_duration, actual_duration)
        # Check onset time
        assert np.isclose(int(note.start * 1000), expected_onset, atol=0.000), f"Onset mismatch at index {i}: expected {expected_onset}, got {int(note.start * 1000)}"

        # Check duration
        # 
        assert np.isclose(actual_duration, expected_duration, atol=0.000), f"Duration mismatch at index {i}: expected {expected_duration}, got {actual_duration}"

        # Check pitch
        assert note.pitch == expected_pitch, f"Pitch mismatch at index {i}: expected {expected_pitch}, got {note.pitch}"

    print("All tests passed!")

# Example usage

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input text file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the file")
    args = parser.parse_args()

    
    output_dir = "/".join(args.output_path.split("/")[:-1])
    print("input:", args.input_path)
    print("output directory:", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("Converting...")
    text_to_midi(args.input_path, args.output_path)
    
    
    print(f"MIDI file saved as {args.output_path}")
    # print("testing...")
    # test_midi_conversion(args.input_path, args.output_path)
