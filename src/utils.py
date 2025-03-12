import json
from fractions import Fraction
from typing import List
import music21
import os

from const_tokens import *
from .note.Note import Note

def serialize_json(obj, indent=4, current_indent=0):
    """
    Recursively serialize JSON object with indentation,
    keeping lists on a single line.
    """
    spaces = ' ' * indent
    current_spaces = ' ' * current_indent
    if isinstance(obj, dict):
        items = []
        for key, value in obj.items():
            serialized_value = serialize_json(value, indent, current_indent + indent)
            items.append(f'{current_spaces}{spaces}"{key}": {serialized_value}')
        return '{\n' + ',\n'.join(items) + f'\n{current_spaces}' + '}'
    elif isinstance(obj, list) or isinstance(obj, tuple):
        # Serialize list on a single line
        serialized_items = [json.dumps(str(item)) for item in obj]
        return '[' + ', '.join(serialized_items) + ']'
    else:
        return json.dumps(str(obj))
    
def get_token_attribute(token, attribute):
    key = {
        "note_length": 0,
        "is_dotted": 1,
        "is_triplet": 2,
        "is_fermata": 3,
        "is_staccato": 4,
        "tied_forward": 5,
        "is_rest": 6,
    }
    return token[key[attribute]]

def get_note_and_length_to_token_id_dicts(id_to_token):
    
    note_length_to_token_ids = {}
    rest_length_to_token_ids = {}
    
    
    for token_id, token in id_to_token.items():
        if token in CONST_TOKENS:
            continue
        curr_len = token.get_len()
        
        if token.is_rest:
            rest_length_to_token_ids[curr_len] = rest_length_to_token_ids.get(curr_len, []) + [token_id]
        elif not token.tied_forward:
            note_length_to_token_ids[curr_len] = note_length_to_token_ids.get(curr_len, []) + [token_id]
            
    # now we need to consider the case of tied notes or multiple consecutive rests
    # we'll only consider one tie or two consecutive rests
    # just consider all pairs of (tied note, note) and (rest, rest)
    
    for token_id, token in id_to_token.items():
        if token in CONST_TOKENS:
            continue
        if token.tied_forward:
            note1_length = token.get_len()
            for token2_id, token2 in id_to_token.items():
                # don't want second token to be constant, rest, or tied forward
                if token2 in CONST_TOKENS or token2.is_rest or token2.tied_forward:
                    continue
                token2_length = token2.get_len()
                curr_len = note1_length + token2_length
                note_length_to_token_ids[curr_len] = note_length_to_token_ids.get(curr_len, []) + [(token_id, token2_id)]
                
        elif token.is_rest:
            rest1_length = token.get_len()
            for rest2_id, token2 in id_to_token.items():
                if token2 in CONST_TOKENS or not token2.is_rest:
                    continue
                rest2_length = token2.get_len()
                curr_len = rest1_length + rest2_length
                rest_length_to_token_ids[curr_len] = rest_length_to_token_ids.get(curr_len, []) + [(token_id, rest2_id)]
                

    return note_length_to_token_ids, rest_length_to_token_ids
        
def convert_note_tuple_list_to_music_xml(note_tuple_list: List, output_dir, pitches=None):
    """
    Convert a list of note tuples to a MusicXML file for visualization
    If pitches is not passed, default to middle C for all notes
    """
    if pitches is not None:
        assert len(note_tuple_list) == len(pitches)
    else:
        pitches = [music21.pitch.Pitch("C4")] * len(note_tuple_list)
        
    pitches = []


def open_processed_data_dir(processed_data_dir):
    with open(os.path.join(processed_data_dir, "token_to_id.json"), "r") as f:
        token_to_id = json.load(f)
        
    with open(os.path.join(processed_data_dir, "id_to_token.json"), "r") as f:
        id_to_token = json.load(f)
        
    with open(os.path.join(processed_data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    # print(token_to_id)
    token_to_id_new = {}
    id_to_token_new = {}
    for key, value in token_to_id.items():
        if not key in CONST_TOKENS:
            key = Note(string=key)
        token_to_id_new[key] = value
        id_to_token_new[value] = key
        
    return token_to_id_new, id_to_token_new, metadata

def mxl_to_note(score_path: str, part_id: int) -> List: # TODO: implement this function
    """
    convert a musicxml file to a list of notes
    MIDI corresponds to musicxml, so we want this split by onsets so this corresponds with that
    output: num onsets x 2, for each cell you can have multiple notes/rests
    output[i][0] and output[i][1] are the notes and rests corresponding to the ith onset
    """
    pass

def decompose_note_sequence(note_sequence: List, token_to_id, id_to_token) -> List:
    """
    Decompose a note sequence into a list of notes and rests
    """
    if note_sequence[0] == token_to_id[START_OF_SEQUENCE_TOKEN]:
        print("yes")
        note_sequence = note_sequence[1:]
    if note_sequence[-1] == token_to_id[END_OF_SEQUENCE_TOKEN]:
        note_sequence = note_sequence[:-1]
    result = [[[], []]]
    detokenized_result = [[[], []]]
    j = 0
    for i in range(len(note_sequence)):
        if note_sequence[i] == token_to_id[START_OF_SEQUENCE_TOKEN]:
            continue
        # increment j if its a new onset: i is a note and i-1 is a rest or a non-tied note, edge case: i > 1 as you have sos at start
        if not id_to_token[note_sequence[i]].is_rest and i >= 1 and (id_to_token[note_sequence[i-1]].is_rest or not id_to_token[note_sequence[i-1]].tied_forward):
            j += 1 
            result += [[[], []]]
            detokenized_result += [[[], []]]
        if not id_to_token[note_sequence[i]].is_rest:
            result[j][0].append(note_sequence[i])
            detokenized_result[j][0].append(id_to_token[note_sequence[i]])
        else:
            result[j][1].append(note_sequence[i])
            detokenized_result[j][1].append(id_to_token[note_sequence[i]])
    return result, detokenized_result
if __name__ == "__main__":
    tmp = [(1, False, False, False, False, False, False), (1, False, False, False, False, False, False), (2, False, False, False, False, False, False), (1, False, False, False, False, False, False), (1, False, False, False, False, False, False), (2, False, False, False, False, False, False)]
    output_dir = "test/"
    convert_note_tuple_list_to_music_xml(tmp, output_dir)
    
    