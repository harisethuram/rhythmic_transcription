import json
from fractions import Fraction
from typing import List
import music21
import os

from const_tokens import *
from .note.Note import Note


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

def flatten_notes(path, note_info_path=None): # TODO: issue with some json where all onsets are put into one string :O
    """
    Takes in path to decomposed notes and flattens them into a single list. If provided note_info_path, also returns list of aligned pitches.
    """
    safe_globals = {"__builtins__": None, "Note": Note}
    with open(path, "r") as f:
        notes = json.load(f)[0]
        notes = eval(notes, safe_globals)

    pitches = [None] * len(notes)
    
    if note_info_path is not None:
        pitches = [i[-1] for i in json.load(open(note_info_path, "r"))]

    results = []
    result_pitches = []
    assert len(notes) == len(pitches), f"len of notes: {len(notes)}, len of pitches: {len(pitches)}"
    for (curr_notes, curr_rests), pitch in zip(notes, pitches):
        curr_notes_Note = [Note(string=str(note)) for note in curr_notes]
        curr_notes_pitches = [pitch] * len(curr_notes_Note)
        curr_rests_Note = [Note(string=str(rest)) for rest in curr_rests]
        curr_rests_pitches = [None] * len(curr_rests_Note)
        results += curr_notes_Note + curr_rests_Note
        result_pitches += curr_notes_pitches + curr_rests_pitches
    # print(f"len of results: {len(results)}, len of result_pitches: {len(result_pitches)}")
    assert len(results) == len(result_pitches), f"len of results: {len(results)}, len of result_pitches: {len(result_pitches)}"
    return results, result_pitches


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
    

def get_tokens_given_length(id_to_token, query, want_rest: bool, cache=None):
    """
    Returns a list of lists, where each sublist contains the token ids of notes or rests (depending on `want_rest`) of a length 'query'. 
    Also returns a cache that you can pass in during future calls to avoid recomputation. Replaces function 'get_note_and_length_to_token_id_dicts'.
    """     
    if cache is not None:
        len_to_tied_forward_notes, len_to_non_tied_notes, len_to_rests, note_length_to_token_ids, rest_length_to_token_ids = cache
    else:
        len_to_tied_forward_notes = {}
        len_to_non_tied_notes = {}
        len_to_rests = {}
        note_length_to_token_ids = {}
        rest_length_to_token_ids = {}
        
        for token_id, token in id_to_token.items():
            if token in CONST_TOKENS:
                continue
            curr_len = token.get_len()
            
            if token.is_rest:
                len_to_rests[curr_len] = len_to_rests.get(curr_len, []) + [token_id]
            else:
                if token.tied_forward:
                    len_to_tied_forward_notes[curr_len] = len_to_tied_forward_notes.get(curr_len, []) + [token_id]
                else:
                    len_to_non_tied_notes[curr_len] = len_to_non_tied_notes.get(curr_len, []) + [token_id]
                    
    # zero case, return empty list
    if query == 0:
        return [], (len_to_tied_forward_notes, len_to_non_tied_notes, len_to_rests, note_length_to_token_ids, rest_length_to_token_ids)
    # try to see if it already exists
    query_dict = rest_length_to_token_ids if want_rest else note_length_to_token_ids
    if query in query_dict:
        return query_dict[query], (len_to_tied_forward_notes, len_to_non_tied_notes, len_to_rests, note_length_to_token_ids, rest_length_to_token_ids)
    
    # if not, we need to compute it and cache it
    tied_forward = len_to_rests if want_rest else len_to_tied_forward_notes
    non_tied = len_to_rests if want_rest else len_to_non_tied_notes
    
    result = non_tied.get(query, []) # initialize with non-tied tokens of query length
    
    for tied_length, tied_token_ids in tied_forward.items():
        if tied_length >= query or tied_length == 0:
            continue
        
        if query - tied_length in non_tied:
            # then, add all pairs 
            for tied_token_id in tied_token_ids:
                for non_tied_token_id in non_tied[query - tied_length]:
                    result.append([tied_token_id, non_tied_token_id])
    # cache the result
    query_dict[query] = result
    return result, (len_to_tied_forward_notes, len_to_non_tied_notes, len_to_rests, note_length_to_token_ids, rest_length_to_token_ids)
    
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
                note_length_to_token_ids[curr_len] = note_length_to_token_ids.get(curr_len, []) + [[token_id, token2_id]]
                
        elif token.is_rest:
            rest1_length = token.get_len()
            for rest2_id, token2 in id_to_token.items():
                if token2 in CONST_TOKENS or not token2.is_rest:
                    continue
                rest2_length = token2.get_len()
                curr_len = rest1_length + rest2_length
                rest_length_to_token_ids[curr_len] = rest_length_to_token_ids.get(curr_len, []) + [[token_id, rest2_id]]


    return note_length_to_token_ids, rest_length_to_token_ids
        
def convert_note_tuple_list_to_music_xml(note_tuple_list: List, output_dir, pitches=None): # TODO
    """
    Convert a list of note tuples to a MusicXML file for visualization
    If pitches is not passed, default to middle C for all notes
    """
    if pitches is not None:
        assert len(note_tuple_list) == len(pitches)
    else:
        pitches = [music21.pitch.Pitch("C4")] * len(note_tuple_list)
        
    pitches = []
    return None


def open_processed_data_dir(processed_data_dir):
    """
    Does the typecasting to help with getting the tokenizer dictionaries from the output of kern_processer.py
    """
    
    with open(os.path.join(processed_data_dir, "token_to_id.json"), "r") as f:
        token_to_id = json.load(f)
        
    with open(os.path.join(processed_data_dir, "id_to_token.json"), "r") as f:
        id_to_token = json.load(f)
    
    metadata = None
    
    if os.path.exists(os.path.join(processed_data_dir, "metadata.json")):   
        with open(os.path.join(processed_data_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
    # print(token_to_id)
    token_to_id_new = {}
    id_to_token_new = {}
    for key, value in token_to_id.items():
        if not key in CONST_TOKENS:
            key = Note(string=key)
        token_to_id_new[key] = int(value)
        id_to_token_new[int(value)] = key
        
    return token_to_id_new, id_to_token_new, metadata

def mxl_to_note(score_path: str, part_id: int) -> List: # TODO: implement this function
    """
    convert a musicxml file to a list of notes
    MIDI corresponds to musicxml, so we want this split by onsets so this corresponds with that
    output: num onsets x 2, for each cell you can have multiple notes/rests
    output[i][0] and output[i][1] are the notes and rests corresponding to the ith onset
    """
    pass

def decompose_note_sequence(note_sequence: List, token_to_id, id_to_token) -> List: # TODO: handle unks
    """
    Decompose a note sequence into a list of notes and rests
    """
    assert note_sequence[0] == token_to_id[START_OF_SEQUENCE_TOKEN], "Note sequence must start with START_OF_SEQUENCE_TOKEN"
    # assert note_sequence[-1] == token_to_id[END_OF_SEQUENCE_TOKEN], "Note sequence must end with END_OF_SEQUENCE_TOKEN"
    print(note_sequence)
    if note_sequence[0] == token_to_id[START_OF_SEQUENCE_TOKEN]:
        # print("yes")
        note_sequence = note_sequence[1:]
    if note_sequence[-1] == token_to_id[END_OF_SEQUENCE_TOKEN]:
        note_sequence = note_sequence[:-1]
    result = [[[], []]]
    detokenized_result = [[[], []]]
    # print("note:", note_sequence)
    # print(note_sequence)
    # print(token_to_id)
    # print(id_to_token)
    # input()
    j = 0
    for i in range(len(note_sequence)):
        curr_tok = id_to_token[note_sequence[i]]
        prev_tok = id_to_token[note_sequence[i-1]]
        # print("m:", note_sequence[i])
        # input()
        # increment j if its a new onset: i is a note and i-1 is a rest or a non-tied note, edge case: i > 1 as you have sos at start
        
        # if curr_tok == UNKNOWN_TOKEN:
        #     j += 1
        #     result += [[token_to_id[UNKNOWN_TOKEN]], [token_to_id[UNKNOWN_TOKEN]]]
        #     detokenized_output += [[UNKNOWN_TOKEN], UNKNOWN_TOKEN]
        #     continue 
        try:
            if not curr_tok.is_rest and i >= 1 and (prev_tok.is_rest or (not prev_tok.tied_forward and not prev_tok.is_rest)):
                j += 1 
                result += [[[], []]]
                detokenized_result += [[[], []]]
            
            #     print(curr_tok)
            #     input()
            if  not curr_tok.is_rest:
                result[j][0].append(note_sequence[i])
                detokenized_result[j][0].append(curr_tok)
            else:
                result[j][1].append(note_sequence[i])
                detokenized_result[j][1].append(curr_tok)
        except:
            print(curr_tok)
            raise ValueError("Invalid token in note sequence: " + str(curr_tok))
    return result, detokenized_result

def decompose_note_sequence_notes(note_sequence: List, token_to_id, id_to_token) -> List: # TODO: handle unks
    """
    Decompose a note sequence into a list of notes and rests
    """
    
    if note_sequence[0] == START_OF_SEQUENCE_TOKEN:
        # print("yes")
        note_sequence = note_sequence[1:]
    if note_sequence[-1] == END_OF_SEQUENCE_TOKEN:
        note_sequence = note_sequence[:-1]
    result = [[[], []]]
    detokenized_result = [[[], []]]
    # print("note:", note_sequence)
    # print(note_sequence)
    # print(token_to_id)
    # print(id_to_token)
    # input()
    j = 0
    for i in range(len(note_sequence)):
        # print(note_sequence[i])
        if note_sequence[i] == START_OF_SEQUENCE_TOKEN:
            continue
        # increment j if its a new onset: i is a note and i-1 is a rest or a non-tied note, edge case: i > 1 as you have sos at start
        curr_tok = note_sequence[i]
        prev_tok = note_sequence[i-1]
        # if curr_tok == UNKNOWN_TOKEN:
        #     j += 1
        #     result += [[token_to_id[UNKNOWN_TOKEN]], [token_to_id[UNKNOWN_TOKEN]]]
        #     detokenized_output += [[UNKNOWN_TOKEN], UNKNOWN_TOKEN]
        #     continue 
        try:
            if not curr_tok.is_rest and i >= 1 and (prev_tok.is_rest or (not prev_tok.tied_forward and not prev_tok.is_rest)):
                j += 1 
                result += [[[], []]]
                detokenized_result += [[[], []]]
        except:
            print(curr_tok)
            input()
        if not curr_tok.is_rest:
            # result[j][0].append(note_sequence[i])
            detokenized_result[j][0].append(curr_tok)
        else:
            # result[j][1].append(note_sequence[i])
            detokenized_result[j][1].append(curr_tok)
    return detokenized_result

def convert_alignment(alignment):
    """
    alignment is a list of (i1, i2) pairs,
      where i1 is an index in sequence 1,
            i2 is an index in sequence 2.
    We return a dictionary such that:
      - if i1 maps to multiple i2's (one-to-many), store final_dict[i1] = (i2_1, i2_2, ...)
      - if multiple i1's map to the same i2 (many-to-one), store final_dict[(i1_1, i1_2, ...)] = i2
      - otherwise normal one-to-one: final_dict[i1] = i2
    Order is preserved based on the order in which i1 first appears in `alignment`.
    """

    # 1) Build forward and backward maps to gather alignments
    from collections import OrderedDict

    forward_map = OrderedDict()  # i1 -> list of i2
    backward_map = {}            # i2 -> list of i1

    for i1, i2 in alignment:
        # Forward map: preserve the order of i1 as it first appears
        if i1 not in forward_map:
            forward_map[i1] = []
        forward_map[i1].append(i2)

        # Backward map: just collect i1 in a list
        if i2 not in backward_map:
            backward_map[i2] = []
        backward_map[i2].append(i1)

    # 2) Build the final dictionary, preserving order of i1 as in forward_map
    used_i1 = set()
    used_i2 = set()
    final_dict = OrderedDict()

    # Helper to keep track of the order in which i1's first appeared
    i1_in_order = list(forward_map.keys())

    for i1 in i1_in_order:
        # Skip if this i1 has already been used in a many-to-one
        if i1 in used_i1:
            continue

        i2_list = forward_map[i1]

        # --- One-to-many case ---
        if len(i2_list) > 1:
            # i1 maps to multiple i2
            # => final_dict[i1] = tuple of i2s
            final_dict[tuple([i1])] = tuple(i2_list)

            # Mark i1, and all the i2s, as used so we don't re-map them
            used_i1.add(i1)
            for x in i2_list:
                used_i2.add(x)

        else:
            # There's exactly one i2 for this i1
            i2_single = i2_list[0]

            # If that i2 is already used, skip or ignore (depends on your data assumptions).
            if i2_single in used_i2:
                continue

            # Check how many i1's map to this i2
            i1_list_for_that_i2 = backward_map[i2_single]

            # Filter out any i1 that's already used
            i1_list_for_that_i2 = [x for x in i1_list_for_that_i2 if x not in used_i1]

            # --- Many-to-one case ---
            if len(i1_list_for_that_i2) > 1:
                # multiple i1's map to the same i2 => key should be a tuple of those i1's, value is i2
                # but preserve the order in which these i1's appeared in the alignment
                i1_positions = {v: idx for idx, v in enumerate(i1_in_order)}
                i1_list_for_that_i2.sort(key=lambda x: i1_positions[x])  # stable sort by appearance
                key = tuple(i1_list_for_that_i2)
                final_dict[key] = tuple([i2_single])

                # Mark them all used
                for x in i1_list_for_that_i2:
                    used_i1.add(x)
                used_i2.add(i2_single)

            else:
                # --- Normal one-to-one ---
                final_dict[tuple([i1])] = tuple([i2_single])
                used_i1.add(i1)
                used_i2.add(i2_single)

    return final_dict



if __name__ == "__main__":
    # tmp = [(1, False, False, False, False, False, False), (1, False, False, False, False, False, False), (2, False, False, False, False, False, False), (1, False, False, False, False, False, False), (1, False, False, False, False, False, False), (2, False, False, False, False, False, False)]
    # output_dir = "test/"
    # # convert_note_tuple_list_to_music_xml(tmp, output_dir)
    id_to_token = json.load(open("processed_data/test_all/id_to_token.json", "r"))
    print(get_note_and_length_to_token_id_dicts(id_to_token))
    
    