import music21
import json
from typing import List, Dict
import numpy as np

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

def tie_chains_from_lists(ties, pitches):
    """
    Given:
       ties[i] is one of None, 'start', 'continue', 'stop'
       pitches[i] is the pitch (could be a music21.pitch.Pitch or any comparable label)
    Return:
       A list of tie-chains, where each chain is a list of integer indexes into ties/pitches.
    """

    # This dict maps pitch -> a stack of "active" chains.
    # Each active chain is just a list of indexes.
    active_chains = defaultdict(list)

    # This will hold our final list of completed tie-chains
    completed_chains = []

    for i, (tie_type, pitch) in enumerate(zip(ties, pitches)):
        # Ignore notes that aren't tied
        if tie_type is None:
            continue

        # Make sure pitch has a stack in active_chains
        # (defaultdict(list) does this automatically)
        chain_stack = active_chains[pitch]

        if tie_type == 'start':
            # Create a new chain for this pitch
            chain_stack.append([i])

        elif tie_type == 'continue':
            # Append this note index to the last chain on the stack
            if not chain_stack:
                # No chain to continue? Gracefully handle by starting a new chain anyway
                chain_stack.append([i])
            else:
                chain_stack[-1].append(i)

        elif tie_type == 'stop':
            # Complete the last chain on the stack
            if not chain_stack:
                # Edge case: got 'stop' but no active chain. 
                # We can either ignore or treat it as a single-note chain:
                completed_chains.append([i])
            else:
                chain = chain_stack.pop()
                chain.append(i)
                completed_chains.append(chain)
        # print(active_chains)

    # If any chains never received a 'stop', you can decide whether to keep them as incomplete
    # or ignore them. Here we’ll just finalize them as-is:
    for pitch, stack in active_chains.items():
        for chain in stack:
            completed_chains.append(chain)

    return completed_chains

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
    
    # we need these three lists to check for polyphony
    bar_numbers = [] # Bars that have polyphony are discarded and the part is spliced on these bars
    pitches = [] # in case of ties forward, if consecutive notes don't have the same pitch, there must be polyphony
    offsets = [] # if two notes are at the same offset and their lengths are different, there must be polyphony
    ties = []
    
    # once we have all the information, we want to sort all by offset, then check for polyphony. Also can do assertion for bar numbers being non-decreasing.
    
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
            if element.tie.type == "start" or element.tie.type == "continue":
                curr_note.tied_forward = True
            ties.append(element.tie.type)
        else:
            ties.append(None)
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
        bar_number.append(element.measureNumber)
        pitches.append(element.pitch)
        offsets.append(element.offset)
        # offsets[element.offset] = offsets.get(element.offset, set()) | {i}
        # if len(offsets[element.offset]) > 1:
        #     if debug:
        #         print("Polyphony detected!")
        #         print(f"Notes at {element.offset}:", offsets[element.offset])
        #         print("Bar number:", element.measureNumber)
        #     return None
    # convert offsets to list of tuples
    
    # sort all by offset
    argsort = np.argsort(offsets)
    rhythms_and_expressions = [rhythms_and_expressions[i] for i in argsort]
    bar_numbers = [bar_numbers[i] for i in argsort]
    pitches = [pitches[i] for i in argsort]
    offsets = [offsets[i] for i in argsort]
    
    polyphonic_idxs = set()
    
    assert all([bar_numbers[i] <= bar_numbers[i+1] for i in range(len(bar_numbers)-1)])
    
    
    # now we want to check for polyphony
    # first check if there are any notes with the same offset and different lengths
    offset_to_note = {}
    offset_to_idxs = {}
    for i, (offset, note) in enumerate(zip(offsets, rhythms_and_expressions)):
        offset_to_note[offset] = offset_to_note.get(offset, set()) | {note}
        offset_to_idxs[offset] = offset_to_note.get(offset, set()) | {i}
        if len(offset_to_note[offset]) > 1:
            polyphonic_idxs |= offset_to_idxs[offset]
            
    # next, check for overlapping notes
    last_end = -1
    last_end_idx = None
    for i, (offset, note) in enumerate(zip(offsets, rhythms_and_expressions)):
        if offset < last_end:
            polyphonic_idxs.add(i)
            polyphonic_idxs.add(last_end_idx)
        if offset + note.get_len() > last_end:
            last_end_idx = i
            last_end = offset + note.get_len()
        last_end = max(last_end, offset + note.get_len())
    
    # check if any of the polyphonic_idxs are tied forward, if so, discard the entire part
    for i in polyphonic_idxs:
        if rhythms_and_expressions[i].tied_forward:
            return []
        
    # otherwise, discard the entire bar where polyphony is detected
    polyphonic_bars = set([bar_numbers[i] for i in polyphonic_idxs])
    # conver to sorted list
    polyphonic_bars = sorted(list(polyphonic_bars))
    
    # now we want to splice the part on these bars
    polyphonic_bar_pointer = 0
    new_rhythms_and_expressions = []
    curr_rhythms_and_expressions = []
    
    for i, (bar_number, note) in enumerate(zip(bar_numbers, rhythms_and_expressions)):
        if bar_number == polyphonic_bars[polyphonic_bar_pointer]:
            if not curr_rhythms_and_expressions:
                continue
            new_rhythms_and_expressions.append(curr_rhythms_and_expressions)
            curr_rhythms_and_expressions = []
            polyphonic_bar_pointer += 1
        else:
            curr_rhythms_and_expressions.append(note)
            
    if curr_rhythms_and_expressions:
        new_rhythms_and_expressions.append(curr_rhythms_and_expressions)
    
    for rhythms_and_expressions in new_rhythms_and_expressions:
        rhythms_and_expressions.append(END_OF_SEQUENCE_TOKEN)
          
    return new_rhythms_and_expressions
    
    
    # finally, check for overlaps caused by tied notes. How can this edge case happen? If n1 of pitch p1 is tied to n2 of pitch p1, and there is n3 of pitch p3 at the same offset as n2, then there is polyphony. 
    # Are there any other edge cases? i don't think so. This is quite complex actually lol, so maybe just discard the entire part if there is polyphony and tie. 
    # first get a list of all tied note chains
    # tied_note_chains = tie_chains_from_lists(ties, pitches)        
    
    # # now check if these tied chains differ
    # for tied_note_chain in tied_note_chains:
    #     other_notes = [offset_to_idxs[offset[i]] for i in tied_note_chain]
    #     lens = [len(other_note) for other_note in other_notes]
    #     if len(set(lens)) > 1:
    #         # there is polyphony, so we want to add all notes in the tied chain and other shared notes to polyphonic_idxs
    #         pass
    
    
    # offset_list = [(offset, list(notes)[0]) for offset, notes in offsets.items()]
    # offset_list.sort(key=lambda x: x[0])
    
    # we want to check if there is any polyphony caused by overlapping notes i.e. earlier note ends after later note starts
    # for i in range(1, len(offset_list)):
    #     if offset_list[i][0] < offset_list[i-1][0] + offset_list[i-1][1].get_len():
    #         # if debug:
    #         #     print("Polyphony detected!")
    #         #     print(f"Notes at {offset_list[i][0]}:", offset_list[i][1])
    #         #     print(f"Notes at {offset_list[i-1][0]}:", offset_list[i-1][1])
    #         #     print("Bar number:", part.flat.notesAndRests[i].measureNumber)
    #         return None
        
    # if debug:
    #     print("Rhythms and expressions:", offsets)


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