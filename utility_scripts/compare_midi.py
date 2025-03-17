import pretty_midi

def compare_midi_files(midi_path_1, midi_path_2, tolerance=1e-5):
    """
    Compare two MIDI files by checking the onsets (start times),
    offsets (end times), and lengths of corresponding notes.
    
    Parameters
    ----------
    midi_path_1 : str
        Path to the first MIDI file.
    midi_path_2 : str
        Path to the second MIDI file.
    tolerance : float
        A small threshold to determine when two times
        or lengths are considered "different."

    Returns
    -------
    List[dict]
        A list of dictionaries, each describing a difference. Each dictionary has:
          - 'instrument_index': Index of instrument in the MIDI files
          - 'note_index': Index of the note within that instrument
          - 'onset_1': Onset time of the note in first MIDI
          - 'onset_2': Onset time of the note in second MIDI
          - 'offset_1': Offset time of the note in first MIDI
          - 'offset_2': Offset time of the note in second MIDI
          - 'length_1': Length of the note in first MIDI
          - 'length_2': Length of the note in second MIDI
    """
    
    # Load both MIDI files
    pm1 = pretty_midi.PrettyMIDI(midi_path_1)
    pm2 = pretty_midi.PrettyMIDI(midi_path_2)
    
    differences = []
    # Compare up to the smaller number of instruments
    min_insts = min(len(pm1.instruments), len(pm2.instruments))
    
    
    for i in range(min_insts):
        inst1 = pm1.instruments[i]
        inst2 = pm2.instruments[i]
        
        # Compare up to the smaller number of notes
        min_notes = min(len(inst1.notes), len(inst2.notes))
        # print(min_notes, len(inst1.notes), len(inst2.notes))
        
        for j in range(min_notes):
            note1 = inst1.notes[j]
            note2 = inst2.notes[j]
            
            # Extract onset, offset, and length
            onset_1, onset_2 = note1.start, note2.start
            offset_1, offset_2 = note1.end, note2.end
            length_1 = offset_1 - onset_1
            length_2 = offset_2 - onset_2
            
            # Check if they differ beyond the given tolerance
            if (abs(onset_1 - onset_2) > tolerance or
                abs(offset_1 - offset_2) > tolerance or
                abs(length_1 - length_2) > tolerance):
                
                # Record the difference
                differences.append({
                    'instrument_index': i,
                    'note_index': j,
                    'onset_1': onset_1,
                    'onset_2': onset_2,
                    'offset_1': offset_1,
                    'offset_2': offset_2,
                    'length_1': length_1,
                    'length_2': length_2
                })
                
        # If the number of notes differs, you might want to capture that too:
        if len(inst1.notes) != len(inst2.notes):
            differences.append({
                'instrument_index': i,
                'note_mismatch': True,
                'notes_in_first': len(inst1.notes),
                'notes_in_second': len(inst2.notes)
            })
    
    # If the number of instruments differs, capture that too:
    if len(pm1.instruments) != len(pm2.instruments):
        differences.append({
            'instrument_mismatch': True,
            'instruments_in_first': len(pm1.instruments),
            'instruments_in_second': len(pm2.instruments)
        })
    
    return differences

if __name__ == "__main__":
    # Example usage:
    file_1 = "data/aligned/URMP/val/27_King_vn_vn_va_sax/Sco_27_King_vn_vn_va_sax.mid"
    file_2 = "misc/Sco_27_King_vn_vn_va_sax.mid"
    diffs = compare_midi_files(file_1, file_2)
    
    if not diffs:
        print("No differences found.")
    else:
        for diff in diffs:
            print(diff)
            input()