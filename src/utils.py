import json
from fractions import Fraction

from const_tokens import *

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
        serialized_items = [json.dumps(item) for item in obj]
        return '[' + ', '.join(serialized_items) + ']'
    else:
        return json.dumps(obj)
    
def get_token_attribute(token, attribute):
    key = {
        "note_length": 0,
        "is_dotted": 1,
        "is_triplet": 2,
        "is_fermata": 3,
        "is_staccato": 4,
        "is_tied_forward": 5,
        "is_rest": 6,
    }
    return token[key[attribute]]

def get_note_and_length_to_token_id_dicts(id_to_token):
    def compute_length(note):
        return Fraction(get_token_attribute(note, "note_length")) * (Fraction(3, 2) if get_token_attribute(note, "is_dotted") else 1) * (Fraction(2, 3) if get_token_attribute(note, "is_triplet") else 1)
    
    note_length_to_token_ids = {}
    rest_length_to_token_ids = {}
    
    for token_id, token in id_to_token.items():
        if token in CONST_TOKENS:
            continue
        curr_len = compute_length(token)
        
        if get_token_attribute(token, "is_rest"):
            rest_length_to_token_ids[curr_len] = rest_length_to_token_ids.get(curr_len, []) + [token_id]
        elif not get_token_attribute(token, "is_tied_forward"):
            note_length_to_token_ids[curr_len] = note_length_to_token_ids.get(curr_len, []) + [token_id]
            
    # now we need to consider the case of tied notes or multiple consecutive rests
    # we'll only consider one tie or two consecutive rests
    # just consider all pairs of (tied note, note) and (rest, rest)
    
    for token_id, token in id_to_token.items():
        if token in CONST_TOKENS:
            continue
        if get_token_attribute(token, "is_tied_forward"):
            note1_length = compute_length(token)
            for token2_id, token2 in id_to_token.items():
                # don't want second token to be constant, rest, or tied forward
                if token2 in CONST_TOKENS or get_token_attribute(token2, "is_rest") or get_token_attribute(token2, "is_tied_forward"):
                    continue
                token2_length = compute_length(token2)
                curr_len = note1_length + token2_length
                note_length_to_token_ids[curr_len] = note_length_to_token_ids.get(curr_len, []) + [(token_id, token2_id)]
                
        elif get_token_attribute(token, "is_rest"):
            rest1_length = compute_length(token)
            for rest2_id, token2 in id_to_token.items():
                if token2 in CONST_TOKENS or not get_token_attribute(token2, "is_rest"):
                    continue
                rest2_length = compute_length(token2)
                curr_len = rest1_length + rest2_length
                rest_length_to_token_ids[curr_len] = rest_length_to_token_ids.get(curr_len, []) + [(token_id, rest2_id)]
                

    return note_length_to_token_ids, rest_length_to_token_ids
        
        
              
    