import music21
import json
from typing import List, Dict

from const_tokens import *
from ..note.Note import Note

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

def get_rhythms_and_expressions(part, want_barlines: bool=False, no_expressions: bool=True, want_leading_rests: bool=False, debug=False) -> List:
    """
    Get the rhythms and expressions for a given part 
    part: music21.stream.Part
    want_barlines: do you want barlines in the output
    no_expressions: whether to include expressions
    want_leading_rests: whether to include leading rest
    returns list of Note objects, or none if the part is invalid (i.e. has notes of different lengths with same offset - polyphony)
    """
    # get first measure number
    for element in part.flat.notesAndRests:
        current_measure = element.measureNumber
        break

    # Iterate through all elements in the part

    rhythms_and_expressions = [START_OF_SEQUENCE_TOKEN]
    num_notes_seen = 0
    offsets = {} # offset to note

    for i, element in enumerate(part.flat.notesAndRests):
        
        if current_measure != element.measureNumber and want_barlines:
            rhythms_and_expressions.append(BARLINE_TOKEN)
            current_measure = element.measureNumber
            
        duration = analyze_duration(element.duration)
        
        curr_note = Note(
            duration=duration[0],
            dotted=duration[2],
            triplet=duration[3],
            fermata=False, 
            staccato=False,
            tied_forward=False,
            is_rest=False,
        )

        # check if note is tied
        if element.tie:
            if element.tie.type == "start":
                curr_note.tied_forward = True
        if not no_expressions:
            # check if note is staccato
            for articulation in element.articulations:
                if isinstance(articulation, music21.articulations.Staccato):
                    curr_note.staccato = True
                    break
            
            # check if note has fermata
            if element.expressions:
                for expression in element.expressions:
                    if isinstance(expression, music21.expressions.Fermata):
                        curr_note.fermata = True
                        break

        # check if note is a rest
        if isinstance(element, music21.note.Rest):
            curr_note.is_rest = True
        else:
            num_notes_seen += 1
            
        if num_notes_seen == 0 and not want_leading_rests and isinstance(element, music21.note.Rest):
            # print("Skipping leading rest", num_notes_seen, curr_note, i)
            # input()
            continue
        
        rhythms_and_expressions.append(curr_note)
        offsets[element.offset] = offsets.get(element.offset, set()) | {curr_note}
        if len(offsets[element.offset]) > 1:
            if debug:
                print("Polyphony detected!")
                print(f"Notes at {element.offset}:", offsets[element.offset])
                print("Bar number:", element.measureNumber)
            return None
    # convert offsets to list of tuples
    offset_list = [(offset, list(notes)[0]) for offset, notes in offsets.items()]
    offset_list.sort(key=lambda x: x[0])
    
    # we want to check if there is any polyphony caused by overlapping notes i.e. earlier note ends after later note starts
    for i in range(1, len(offset_list)):
        if offset_list[i][0] < offset_list[i-1][0] + offset_list[i-1][1].get_len():
            if debug:
                print("Polyphony detected!")
                print(f"Notes at {offset_list[i][0]}:", offset_list[i][1])
                print(f"Notes at {offset_list[i-1][0]}:", offset_list[i-1][1])
                print("Bar number:", part.flat.notesAndRests[i].measureNumber)
            return None
    # if debug:
    #     print("Rhythms and expressions:", offsets)
    rhythms_and_expressions.append(END_OF_SEQUENCE_TOKEN)
    return rhythms_and_expressions

def get_tuple(duration, dotted=False, triplet=False, fermata=False, staccato=False, tied_forward=False, is_rest=False):
    return (duration, dotted, triplet, fermata, staccato, tied_forward, is_rest)


def get_base_tokenizer_dicts(want_barlines=False, no_expressions=True):
    # note properties: duration, dotted, triplet, fermata, staccato, tied_forward, is_rest
    durations = [2**i for i in range(-4, 4)]
    properties = [False, True]
    articulation_properties = [False] if no_expressions else [False, True]
    
    # Generate all possible combinations of note properties
    token_to_id = {}
    
    # Add special tokens for padding and unknown tokens
    count = 0
    # pad token
    token_to_id[PADDING_TOKEN] = count
    count += 1
    
    # unknown token
    token_to_id[UNKNOWN_TOKEN] = count
    count += 1
    
    # start of sequence token
    token_to_id[START_OF_SEQUENCE_TOKEN] = count
    count += 1
    
    # end of sequence token
    token_to_id[END_OF_SEQUENCE_TOKEN] = count
    count += 1
    
    # barline token
    if want_barlines:
        token_to_id[BARLINE_TOKEN] = count
        count += 1
    
    id_to_token = {v: k for k, v in token_to_id.items()}
    
    return token_to_id, id_to_token

def tokenize(notes: List, token_to_id: Dict) -> List:
    """
    Tokenizes the given notes using the given token_to_id dictionary.
    
    Parameters:
    notes (list): List of notes to tokenize.
    token_to_id (dict): Dictionary mapping tokens to their corresponding IDs.
    
    Returns:
    list: List of tokenized notes.
    """
    tokenized_notes = []
    for note in notes:
        if note in token_to_id:
            tokenized_notes.append(token_to_id[note])
        else:
            tokenized_notes.append(token_to_id[UNKNOWN_TOKEN])
    return tokenized_notes