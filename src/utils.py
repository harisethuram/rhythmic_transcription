import json

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
    note_length_to_token_ids = {}
    rest_length_to_token_ids = {}
    
    for token_id, token in id_to_token.items():
        curr_len = get_token_attribute(token, "note_length") * (1.5 if get_token_attribute(token, "is_dotted") else 1) * (2/3 if get_token_attribute(token, "is_triplet") else 1)
        
        if get_token_attribute(token, "is_rest"):
            rest_length_to_token_ids[curr_len] = rest_length_to_token_ids.get(curr_len, []) + [token_id]
        else:
            note_length_to_token_ids[curr_len] = note_length_to_token_ids.get(curr_len, []) + [token_id]
            
    return note_length_to_token_ids, rest_length_to_token_ids
        
        
              
    