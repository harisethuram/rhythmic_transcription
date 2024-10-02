from music21 import converter, note
import music21
from src.preprocess.utils import analyze_duration
import argparse
import os
import tqdm

def get_rhythms_and_expressions(part):
    current_measure = None

    # Iterate through all elements in the part
    count = 0

    rhythms_and_expressions = []

    for element in part.flat.notesAndRests:
        
        duration = analyze_duration(element.duration)
        curr_note = {
            "duration": duration[0],
            "dotted": duration[2],
            "triplet": duration[3],
            "fermata": False, 
            "staccato": False,
            "tied_forward": False,
            "is_rest": False,
            "measure": current_measure
        }

        # check if note is tied
        if element.tie:
            if element.tie.type == "start":
                curr_note["tied_forward"] = True

        # check if note is staccato
        for articulation in element.articulations:
            if isinstance(articulation, music21.articulations.Staccato):
                curr_note["staccato"] = True
                break
        
        # check if note has fermata
        if element.expressions:
            for expression in element.expressions:
                if isinstance(expression, music21.expressions.Fermata):
                    curr_note["fermata"] = True
                    break

        # check if note is a rest
        if isinstance(element, music21.note.Rest):
            curr_note["is_rest"] = True
       
        rhythms_and_expressions.append(curr_note)
            
        if count > 40:
            break
        count += 1

    return rhythms_and_expressions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts a directory of kern files into a list of tokens.")
    parser.add_argument("--input_dir", type=str, help="The directory containing the kern files.")
    parser.add_argument("--output_dir", type=str, help="The file to write the tokenized output to.")
    args = parser.parse_args()

    # get the rhyths and expressions for every part for every file in the input directory
    print("Getting rhythms and expressions for all parts in all files...")
    all_rhythms_and_expressions = {}
    count = 0
    for file in tqdm(os.listdir(args.input_dir)):
        if count > 10:
            break
        count += 1
        # print(file)
        parts = converter.parse(os.path.join(args.input_dir, file)).parts
        for i, part in enumerate(parts):
            all_rhythms_and_expressions[(file, i)] = get_rhythms_and_expressions(part)
    
    # get all unique note tokens
    note_tokens = []
    for rhythms_and_expressions in all_rhythms_and_expressions.values():
        for note in rhythms_and_expressions:
            note_tokens.append((note["duration"], note["dotted"], note["triplet"], note["fermata"], note["staccato"], note["tied_forward"], note["is_rest"]))
    
    unique_note_tokens = list(set(note_tokens))

    # assign token ids to each unique note token
    token_to_id = {}
    id_to_token = {}
    token_to_id[(0, False, False, False, False, False, False)] = 0 # barline
    id_to_token[0] = (0, False, False, False, False, False, False)
    # pad token
    token_to_id[(-1, False, False, False, False, False, False)] = 1
    id_to_token[1] = (-1, False, False, False, False, False, False)

    for i, token in enumerate(unique_note_tokens):
        token_to_id[token] = i + 1
        id_to_token[i + 1] = token
    
    # write the tokenized output to the output file
    print("Writing tokenized output to file...")
    
