import math
from midiutil import MIDIFile

def freq_to_midi_note_number(freq):
    """Convert frequency in Hz to MIDI note number."""
    midi_note = 69 + 12 * math.log2(freq / 440.0)
    return int(round(midi_note))

def seconds_to_beats(time_in_seconds, tempo):
    """Convert time in seconds to beats based on the tempo."""
    return time_in_seconds * (tempo / 60.0)

# Read the data from the text file
notes = []
with open('data/URMP/val/07_GString_tpt_tbn/Notes_1_tpt_07_GString.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:
            onset_time = float(parts[0])
            pitch_hz = float(parts[1])
            duration = float(parts[2])
            notes.append((onset_time, pitch_hz, duration))

# MIDI file parameters
track    = 0
channel  = 0
time     = 0       # Start time in beats
tempo    = 120     # Beats per minute
volume   = 100     # Volume (0-127)

# Create the MIDIFile object
MyMIDI = MIDIFile(1)  # One track
MyMIDI.addTempo(track, time, tempo)

# Add notes to the MIDI file
for onset_time, pitch_hz, duration in notes:
    beat_time = seconds_to_beats(onset_time, tempo)
    beat_duration = seconds_to_beats(duration, tempo)
    midi_note = freq_to_midi_note_number(pitch_hz)
    MyMIDI.addNote(track, channel, midi_note, beat_time, beat_duration, volume)

# Write the MIDI file to disk
with open('gstring.mid', 'wb') as output_file:
    MyMIDI.writeFile(output_file)
