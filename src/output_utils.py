from music21 import stream, note, tie, meter, clef
from music21 import duration as m21duration
from music21 import pitch as m21pitch
from music21.stream import makeNotation
from typing import List
from math import ceil
import os
from tqdm import tqdm
import re

from music21 import environment
env = environment.UserSettings()
env['lilypondPath'] = '/mmfs1/gscratch/ark/hari/rhythmic_trascription/lilypond-2.25.30/bin/lilypond'


from .note.Note import Note

def note_pitches_to_xml_no_barlines(notes: List[Note], pitches: List[str], output_path: str, num_events_per_measure: int = 16):
    """Convert a list of notes and pitches to a MusicXML file without barlines.
    num_events_per_measure is used to ensure that not all notes go into a single line on the rendered sheet music."""
    # determine total length
    # split notes into measures of num_events_per_measure
    # print(notes[:5])
    use_bass_clef = False
    if pitches:  # avoid division by zero
        middle_c = m21pitch.Pitch('C4').midi
        num_below_c4 = sum(
            1 for p in pitches
            if m21pitch.Pitch(p).midi < middle_c
        )
        use_bass_clef = (num_below_c4 / len(pitches)) > 0.5
        
    is_triplet = [n.triplet for n in notes]
    all_notes = []
    for i in range(0, len(notes), num_events_per_measure):
        all_notes.append(notes[i:i+num_events_per_measure])
        
    all_pitches = []
    for i in range(0, len(pitches), num_events_per_measure):
        all_pitches.append(pitches[i:i+num_events_per_measure])
        
    
    score = stream.Score()
    part = stream.Part()
    
    # if use_bass_clef:
    #     part.insert(0, clef.BassClef())
    # else:
    #     part.insert(0, clef.TrebleClef())
    
    measure_num = 1
    for notes, pitches in tqdm(zip(all_notes, all_pitches), total=len(all_notes), desc="Writing MusicXML"):
        # total_length = sum(note.get_len() for note in notes) * 1/4
        # print(total_length)
        # print(type(total_length))
        # input()
        # total_length = ceil(total_length)
        
        
        m = stream.Measure(number=measure_num)
        
        if measure_num == 1:
            if use_bass_clef:
                m.insert(0, clef.BassClef())
            else:
                m.insert(0, clef.TrebleClef())
        # m.timeSignature = meter.TimeSignature(f'{total_length}')
        # m.timeSignature = meter.TimeSignature(f'{total_length}/4')
        
        for n, pitch in zip(notes, pitches):
            # print(n.get_len(), pitch)
            # input()
            event = n.get_music21_event(pitch)
            m.append(event)
        # print measure
        # print(m.show('text'))  
        # input()
        
        score.append(m)
        measure_num += 1
        
    s = score.flat

    # tf = m21duration.TupletFixer(s)
    # groups = tf.findTupletGroups(incorporateGroupings=True)
    # for g in groups:
    #     tf.fixBrokenTupletDuration(g)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # makeNotation.makeTupletBrackets(score, inPlace=True)
    
    # for i, n in enumerate(score.recurse().notes):
    #     if n.duration.tuplets:
    #         print(i, n, n.duration.quarterLength,
    #             n.duration.tuplets[0].fullName,
    #             n.duration.tuplets[0].type)
    # score.write('musicxml', fp=output_path)
    # output_path = "test/output.ly"
    score.write('lilypond', fp=output_path)
    lilypond_text = Path(output_path).read_text(encoding='utf-8')
    
    transformed_text = transform_lilypond(lilypond_text, is_triplet)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transformed_text)
        
    # pdf now
    os.system(f'lilypond -o {os.path.splitext(output_path)[0]} {output_path}')
    
        
from pathlib import Path

def transform_lilypond(text: str, is_triplet: list[bool]) -> str:
    # Work line-by-line
    lines = text.splitlines(keepends=True)

    # 1) Remove the include line (second line in your example)
    lines = [ln.strip() for ln in lines if not ln.lstrip().startswith(r"\include")]
    # print(lines)
    # print("*" * 50)
    
    # 2) split the \new Voice { note in to two lines
    new_lines = []
    for line in lines:
        if "\\new Voice" in line:
            parts = line.split("{", 1)
            new_lines.append((parts[0] + "{\n \\cadenzaOn").strip())
            new_lines.append(parts[1].strip())
        else:
            new_lines.append(line)
    lines = new_lines
    
    # 3) For each note line, if the corresponding is_triplet is True, wrap it in \tuplet 3/2 { ... }
    # first find the new Voice line
    new_lines = []
    in_voice = False
    triplet_idx = 0
    # print(is_triplet)
    for line in lines:
        if "\\new Voice" in line:
            in_voice = True
            new_lines.append(line)
            # print("entered voice****************")
            continue
        if in_voice and triplet_idx < len(is_triplet):
            if not line.startswith("\\bar"):
                # print("processing line:", line.strip(), "is_triplet:", is_triplet[triplet_idx])
                line = "\\tuplet 3/2 { " + line.strip() + " }" if is_triplet[triplet_idx] else line.strip()
                # print("transformed line:", line)
                triplet_idx += 1
                new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    # print(new_lines)
    return "\n".join(new_lines)

    


# def main():
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="Modify a LilyPond file: remove include, add cadenza, and wrap triplets."
#     )
#     parser.add_argument("input", help="input .ly file")
#     parser.add_argument("output", help="output .ly file")
#     args = parser.parse_args()

#     src = Path(args.input).read_text(encoding="utf-8")
#     transformed = transform_lilypond(src, is_triplet)
#     Path(args.output).write_text(transformed, encoding="utf-8")


# if __name__ == "__main__":
#     main()
