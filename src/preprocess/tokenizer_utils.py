import music21
import json

def analyze_duration(dur):
        """
        Analyzes the given float duration and returns a tuple with:
        - base note value (e.g., 1.0 for a quarter note)
        - dot value (e.g., 0.5 for a dotted note)
        - whether the note is dotted (True/False)
        - whether the note is a triplet (True/False)
        
        Parameters:
        dur (float): The duration in terms of quarter notes.
        
        Returns:
        tuple: (base_value, dot_value, is_dotted, is_triplet)
        """
        base_value = dur.quarterLength
        float_duration = dur.quarterLength
        dot_value = 0
        is_dotted = False
        is_triplet = False
        
        # Define common base note values (in quarter note lengths)
        note_values = [2**i for i in range(-3, 3)] # 4.0, 2.0, 1.0, 0.5, 0.25, 0.125]  # whole, half, quarter, eighth, sixteenth, thirty-second
        
        # Check for triplets first
        triplet_ratio = 2/3
        for value in note_values:
            if abs(float_duration - value * triplet_ratio) < 0.001:
                base_value = value
                is_triplet = True
                is_dotted = False
                dot_value = 0
                return base_value, dot_value, is_dotted, is_triplet
        
        # Check if the duration is a dotted note
        for value in note_values:
            if float_duration == value * 1.5:  # Dotted note
                base_value = value
                dot_value = value / 2
                is_dotted = True
                break
            elif float_duration == value:  # Non-dotted note
                base_value = value
                dot_value = 0
                is_dotted = False
                break

        return base_value, dot_value, is_dotted, is_triplet

def get_rhythms_and_expressions(part, want_barlines=False, no_expressions=True):
    current_measure = 0

    # Iterate through all elements in the part

    rhythms_and_expressions = []

    for element in part.flat.notesAndRests:
        if current_measure != element.measureNumber and want_barlines:
            rhythms_and_expressions.append({
                "duration": 0,
                "dotted": False,
                "triplet": False,
                "fermata": False,
                "staccato": False,
                "tied_forward": False,
                "is_rest": False,
                "pitch": -1
            })
            current_measure = element.measureNumber
            
        duration = analyze_duration(element.duration)
        
        curr_note = {
            "duration": duration[0],
            "dotted": duration[2],
            "triplet": duration[3],
            "fermata": False, 
            "staccato": False,
            "tied_forward": False,
            "is_rest": False,
            # "pitch": eleme
        }
        if isinstance(element, music21.note.Note):
            curr_note["pitch"] = str(element.pitch)
        else:
            curr_note["pitch"] = str(element.pitches)

        # check if note is tied
        if element.tie:
            if element.tie.type == "start":
                curr_note["tied_forward"] = True
        if not no_expressions:
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
        # print(curr_note)
        # input()
        
        rhythms_and_expressions.append(curr_note)

    return rhythms_and_expressions

def get_tuple(duration, dotted=False, triplet=False, fermata=False, staccato=False, tied_forward=False, is_rest=False):
    return (duration, dotted, triplet, fermata, staccato, tied_forward, is_rest)

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
    elif isinstance(obj, list):
        # Serialize list on a single line
        serialized_items = [json.dumps(item) for item in obj]
        return '[' + ', '.join(serialized_items) + ']'
    else:
        return json.dumps(obj)

def tokenizer(want_barlines, no_expressions):
    # note properties: duration, dotted, triplet, fermata, staccato, tied_forward, is_rest
    durations = [2**i for i in range(-4, 4)]
    properties = [False, True]
    articulation_properties = [False] if no_expressions else [False, True]
    
    # Generate all possible combinations of note properties
    token_to_id = {}
    
    # Add special tokens for padding and unknown tokens
    count = 0
    # pad token
    token_to_id[get_tuple(-1)] = count
    count += 1
    
    # unknown token
    token_to_id[get_tuple(-2)] = count
    count += 1
    
    # barline
    if want_barlines:    
        token_to_id[get_tuple(0)] = count
        count += 1
    
    for duration in durations:
        for dotted in properties:
            for triplet in properties:
                for fermata in articulation_properties:
                    for staccato in articulation_properties:
                        for tied_forward in properties:
                            for is_rest in properties:
                                token_to_id[get_tuple(duration, dotted, triplet, fermata, staccato, tied_forward, is_rest)] = count
                                count += 1
    
    id_to_token = {v: k for k, v in token_to_id.items()}
    
    return token_to_id, id_to_token