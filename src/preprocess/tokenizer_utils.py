import music21
import json
from typing import List, Dict
import numpy as np
from fractions import Fraction

from const_tokens import *
from ..note.Note import Note


def dur_to_notes(dur):
    # find if single note fits
    def frac(i):
        return Fraction(1, 2**-i) if i < 0 else Fraction(2**i, 1)
    lowest = -3
    highest = 2
    base_two_notes = [Note(duration=frac(i), dotted=dot, triplet=False) for i in range(lowest, highest+1) for dot in [False, True]]
    triplets = [Note(duration=frac(i), dotted=False, triplet=True) for i in range(lowest, highest+1)]

    tol = frac(lowest) / 12
    notes = base_two_notes + triplets
    for note in notes:
        if abs(note.get_len() - dur) < tol:
            return [note]
    
    # check if dur requires a tie
    first_candidates = sorted(base_two_notes, key=lambda x: -x.get_len())
    second_candidates = sorted(notes, key=lambda x: -x.get_len())
    
    for first in first_candidates:
        if first.get_len() > dur - 7*tol:
            continue
        for second in second_candidates:
            if abs((first.get_len() + second.get_len()) - dur) < tol:
                first.tied_forward = True
                return [first, second]
    return None

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
    if not isinstance(dur, music21.note.Note):
        base_value = dur
        float_duration = dur
    else:   
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

def normalize_chords(part):
    """
    Replace chords in part with constituent notes
    """
    for element in part.recurse():
        if isinstance(element, music21.chord.Chord):
            # get the parent component
            parent = element.activeSite
            # print(element.offset, [element.notes[i].nameWithOctave for i in range(len(element.notes))])
            # print("parent before:")
            # for tmp in parent.recurse():
            # # if isinstance(tmp, music21.note.Note):
            #     print(f"{tmp}->{tmp.offset}", end=",")
            
            # remove element from the parent
            
            
            # get the notes in the chord
            for note in element.notes:
                parent.insert(element.offset, note)
            parent.remove(element)    
            # print("\nparent after:")
            # for tmp in parent.recurse():
            #     # if isinstance(tmp, music21.note.Note):
            #     print(f"{tmp}->{tmp.offset}", end=", ")
            # input()
def normalize_measures(part):
    """
    ensure that measure i + 1 is always one greater than measure i
    """
    # starting_measure = 0
    measures = part.getElementsByClass(music21.stream.Measure)
    starting_measure = measures[0].number
    # print("Starting measure:", starting_measure)
    if not measures:
        return
    if starting_measure is None:
        starting_measure = 1
    for i, measure in enumerate(measures):
        measure.number = starting_measure + i
    
    
def remove_grace_notes(part):
    for element in part.recurse():
        if isinstance(element, music21.note.Note):
            # get the parent component
            parent = element.activeSite
            if element.duration.quarterLength == 0:
                # remove element from the parent
                parent.remove(element)


def convert_part_to_interval_list(ties, pitches, bar_numbers, offsets, rhythms_and_expressions):
    """
    Convert input lists to a list of intervals of tuples of form (start, end, start_bar, end_bar, pitch, [indices])
    Also returns a dictionary of intervals where key is (start, end, start_bar, end_bar) and value is a list of lists of indices corresponding to all notes that make up the interval
    """
    
    intervals = []
    pitch_to_tie_chain = {}
    
    for i, (tie, pitch, bar_number, offset, r_and_e) in enumerate(zip(ties, pitches, bar_numbers, offsets, rhythms_and_expressions)):
        if tie is None:
            intervals.append((offset, offset + r_and_e.get_len(), bar_number, bar_number, pitch, [i]))
            
        else: # in this case we need to check for tie chains
            if tie == "start":
                if pitch in pitch_to_tie_chain and pitch_to_tie_chain[pitch] is not None:
                    print(pitch_to_tie_chain)
                    raise ValueError("Tie start without stop tie in bar", bar_number, "at offset", offset, "with pitch", pitch)
                
                pitch_to_tie_chain[pitch] = [i]
                
            elif tie == "continue":
                if pitch not in pitch_to_tie_chain or pitch_to_tie_chain[pitch] is None:
                    print("ERROR:")
                    print(pitch_to_tie_chain)
                    raise ValueError("Tie continuation without start tie in bar", bar_number, "at offset", offset, "with pitch", pitch)
                
                pitch_to_tie_chain[pitch].append(i)
                
            elif tie == "stop": # this assumes all ties are stopped
                if pitch not in pitch_to_tie_chain or pitch_to_tie_chain[pitch] is None:
                    print("ERROR:")
                    print(pitch_to_tie_chain)
                    raise ValueError("Tie stop without start tie in bar", bar_number, "at offset", offset, "with pitch", pitch)
                
                pitch_to_tie_chain[pitch].append(i)
                curr_tie_chain = pitch_to_tie_chain[pitch]
                pitch_to_tie_chain[pitch] = None
                intervals.append((offsets[curr_tie_chain[0]], offset + r_and_e.get_len(), bar_numbers[curr_tie_chain[0]], bar_number, pitch, curr_tie_chain))
    
    unique_intervals_without_pitch = sorted(list(set([(interval[0], interval[1], interval[2], interval[3]) for interval in intervals])), key=lambda x: x[0])
    intervals = sorted(intervals, key=lambda x: x[0])
    
    interval_dict = {}
    for interval in intervals:
        interval_dict[(interval[0], interval[1], interval[2], interval[3])] = interval_dict.get((interval[0], interval[1], interval[2], interval[3]), []) + [interval[5]]
    
    return intervals, unique_intervals_without_pitch, interval_dict

def get_rhythms_and_expressions(part, want_barlines: bool=False, no_expressions: bool=True, want_leading_rests: bool=False, debug=False, part_name=None) -> List:
    """
    Get the rhythms and expressions for a given part 
    part: music21.stream.Part
    want_barlines: do you want barlines in the output
    no_expressions: whether to include expressions
    want_leading_rests: whether to include leading rest
    returns list of Note objects, or none if the part is invalid (i.e. has notes of different lengths with same offset - polyphony)
    """
    
    # replace any chords with their constituent notes
    normalize_chords(part)
    normalize_measures(part)
    # print("normalized")
    # remove grace notes
    remove_grace_notes(part)
    
    # get first measure number
    for element in part.flat.notesAndRests:
        current_measure = element.measureNumber
        break

    # Iterate through all elements in the part

    rhythms_and_expressions = []
    
    # we need these three lists to check for polyphony
    bar_numbers = [] # Bars that have polyphony are discarded and the part is spliced on these bars
    pitches = [] # in case of ties forward, if consecutive notes don't have the same pitch, there must be polyphony
    offsets = [] # if two notes are at the same offset and their lengths are different, there must be polyphony
    ties = []
    
    error_output = ([], None)
    
    # once we have all the information, we want to sort all by offset, then check for polyphony. Also can do assertion for bar numbers being non-decreasing.
    
    num_notes_seen = 0
    current_measure = 0
    for i, element in enumerate(part.flat.notesAndRests):
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

        if isinstance(element, music21.note.Rest): # if is a rest
            curr_note.is_rest = True
            pitches.append(None)
        elif isinstance(element, music21.chord.Chord): # elif is a chord, separate all notes
            # add each 
            print("Chord detected at bar", element.measureNumber, "with pitches", [n.nameWithOctave for n in element.notes])
            raise ValueError("error in normalize_chords, chord detected")
            # pitches.append(element.sortAscending()[0])
        else: # else is note
            num_notes_seen += 1
            pitches.append(element.pitch.nameWithOctave)

        rhythms_and_expressions.append(curr_note)
        offsets.append(element.offset)
        
        bar_numbers.append(element.measureNumber)
        
    
    check0 = len(bar_numbers) == len(rhythms_and_expressions) == len(pitches) == len(offsets) == len(ties)
    if not check0:
        print("Length mismatch between bar numbers, rhythms and expressions, pitches, offsets, and ties")
        return error_output

    polyphonic_idxs = set()
    
    
    check1 = [(bar_numbers[i] is not None) and (bar_numbers[i+1] is not None) and (bar_numbers[i] <= bar_numbers[i+1]) for i in range(len(bar_numbers)-1)]
    # print("Bar numbers:", bar_numbers)
    if not all(check1):
        print("Error: Bar numbers are not in non-decreasing order:")
        print("Bar numbers:", bar_numbers)
        print(check1.index(False), bar_numbers[check1.index(False)], bar_numbers[check1.index(False)+1], rhythms_and_expressions[check1.index(False)+1], offsets[check1.index(False)+1], pitches[check1.index(False)+1], ties[check1.index(False)+1])
        # raise ValueError("Bar numbers are not in non-decreasing order")
        return error_output
    bar_to_freq = {}
    for i in range(len(bar_numbers)):
        bar_to_freq[bar_numbers[i]] = bar_to_freq.get(bar_numbers[i], 0) + 1

    intervals, unique_intervals_without_pitch, interval_dict = convert_part_to_interval_list(ties, pitches, bar_numbers, offsets, rhythms_and_expressions)
    
    # if multiple lists of indices correspond to the same interval in interval_dict, get rid of all but the first
    # want as output a list of indices that aren't redundant based on their intervals
    new_indices = []
    for interval, info in interval_dict.items():
        new_indices += info[0]
    
    new_indices.sort()
    missings = []
    for i, bar_number in enumerate(bar_numbers):
        if i not in new_indices:
            missings.append(bar_number)
    # print(new_indices)
    # print(missings)
    
    polyphonic_bars = set()
    # check if any of the intervals overlap and if so, add the bar numbers to polyphonic_bars
    farthest = unique_intervals_without_pitch[0]
    for i in range(1, len(unique_intervals_without_pitch)):
        if farthest[1] > unique_intervals_without_pitch[i][0]: # overlap
            polyphonic_bars |= set([j for j in range(unique_intervals_without_pitch[i][2], unique_intervals_without_pitch[i][3]+1)])
            polyphonic_bars |= set([j for j in range(farthest[2], farthest[3]+1)])
            # print(unique_intervals_without_pitch[i], farthest, farthest[1], unique_intervals_without_pitch[i][0])
        
        if unique_intervals_without_pitch[i][1] > farthest[1]:
            farthest = unique_intervals_without_pitch[i]
    # input()    
    
    # return error_output
    # print(set(bar_numbers))
    # print(len(bar_numbers))
    # print("Polyphonic bars:", sorted(list(polyphonic_bars)))
    
    # now that we've accounted for all overlaps, we want to splice
    curr_splice = []
    all_splices = []
    for i, idx in enumerate(new_indices):
        if bar_numbers[idx] in polyphonic_bars:
            if curr_splice:
                all_splices.append(curr_splice)
                curr_splice = []
        else:
            curr_splice.append(idx)
            
    if curr_splice:
        all_splices.append(curr_splice)
    # print("All splices:", all_splices)
    # for splice in all_splices:
    #     print("Splice:", bar_numbers[splice[0]], bar_numbers[splice[-1]])
        
    # remove leading rests
    if not want_leading_rests:
        for splice in all_splices:
            while splice and rhythms_and_expressions[splice[0]].is_rest:
                splice.pop(0)
                
    # create new rhythms and expressions and bar numbers
    all_rhythms_and_expressions = []
    all_bar_numbers = []
    
    for splice in all_splices:
        all_rhythms_and_expressions.append([rhythms_and_expressions[i] for i in splice])
        all_bar_numbers.append([bar_numbers[i] for i in splice])
    
    # add barlines if necessary
    if want_barlines:
        for rhythm_and_expressions, bar_numbers in zip(all_rhythms_and_expressions, all_bar_numbers):
            bar_borders = [i+1 for i in range(len(bar_numbers)-1) if bar_numbers[i] != bar_numbers[i+1]][::-1]
            for i in bar_borders:
                rhythm_and_expressions.insert(i, BARLINE_TOKEN)
    # print("All rhythms and expressions:")
    
    def print_with_barlines(rhythm_and_expressions):
        curr_string = ""
        for i, note in enumerate(rhythm_and_expressions):
            curr_string += str(note) + ", "
            
            if note == BARLINE_TOKEN:
                print("\"" + curr_string + "\"")
                curr_string = ""
        print("\"" + curr_string + "\"")
    
    # for rhythm_and_expressions in all_rhythms_and_expressions:
    #     print("first 20:")
    #     # split print by barline
    #     # curr_string = ""
    #     print(print_with_barlines(rhythm_and_expressions[:20]))
    #     print("last 20:")
    #     print(print_with_barlines(rhythm_and_expressions[-20:]))
    #     # print(rhythm_and_expressions[:20])
    #     # print(rhythm_and_expressions[-20:])
    #     input()
    
    # finally, add start and end of sequence tokens
    for rhythm_and_expressions in all_rhythms_and_expressions:
        rhythm_and_expressions.insert(0, START_OF_SEQUENCE_TOKEN)
        rhythm_and_expressions.append(END_OF_SEQUENCE_TOKEN)
        
    return all_rhythms_and_expressions, sorted(list(polyphonic_bars))
    # input()
    # return error_output
    
    # fix splicing to include all different lists and remove leading rests and add barlines to each if necessary
    
    # # otherwise, discard the entire bar where polyphony is detected
    # polyphonic_bars = set([bar_numbers[i] for i in polyphonic_idxs])
    # # convert to sorted list
    # polyphonic_bars = sorted(list(polyphonic_bars))
    # print("Polyphonic bars:", polyphonic_bars)
    
    # if want_barlines:
    #     rhythms_and_expressions, bar_numbers, offsets = add_barlines(rhythms_and_expressions, bar_numbers, offsets)
    # # return [rhythms_and_expressions]
    # # now we want to splice the part on these bars
    # polyphonic_bar_pointer = 0    
    
    # new_rhythms_and_expressions = []
    # curr_rhythms_and_expressions = []
    # new_bar_numbers = []
    # curr_bar_numbers = []
    # new_pitches = []
    # curr_pitches = []
    # new_offsets = []
    # curr_offsets = []
    # new_ties = []
    # curr_ties = []
    # this is wrong
    # for i, (bar_number)
    
    # for i, (bar_number, note, pitch, offset, tie) in enumerate(zip(bar_numbers, rhythms_and_expressions, pitches, offsets, ties)):
    #     if polyphonic_bar_pointer < len(polyphonic_bars) and bar_number == polyphonic_bars[polyphonic_bar_pointer]:
    #         if not curr_rhythms_and_expressions:
    #             continue
    #         new_rhythms_and_expressions.append(curr_rhythms_and_expressions)
    #         curr_rhythms_and_expressions = []
    #         new_bar_numbers.append(curr_bar_numbers)
    #         curr_bar_numbers = []
    #         new_offsets.append(curr_offsets)
    #         curr_offsets = []
            
    #         polyphonic_bar_pointer += 1
    #         print("Splicing on bar", bar_number, polyphonic_bar_pointer)
    #     else:
    #         curr_rhythms_and_expressions.append(note)
    #         curr_bar_numbers.append(bar_number)
    #         curr_offsets.append(offset)            
            
            
    # if curr_rhythms_and_expressions:
    #     new_rhythms_and_expressions.append(curr_rhythms_and_expressions)
    #     new_bar_numbers.append(curr_bar_numbers)
    #     new_offsets.append(curr_offsets)

    # new_rhythms_and_expressions = [merge_notes(rhythms_and_expressions, offsets) for rhythms_and_expressions, offsets in zip(new_rhythms_and_expressions, new_offsets)]

    # # we also want to remove any leading rests
    # if not want_leading_rests:
    #     for i, rhythms_and_expressions in enumerate(new_rhythms_and_expressions):
    #         first_note_idx = 0
    #         for j, note in enumerate(rhythms_and_expressions):
    #             if note != BARLINE_TOKEN and note.is_rest == False:
    #                 first_note_idx = j
    #                 break
    #         new_rhythms_and_expressions[i] = rhythms_and_expressions[first_note_idx:]    
    
    # for rhythms_and_expressions in new_rhythms_and_expressions:
    #     # add start of sequence token
    #     rhythms_and_expressions.insert(0, START_OF_SEQUENCE_TOKEN)
    #     # add end of sequence token
    #     rhythms_and_expressions.append(END_OF_SEQUENCE_TOKEN)
    
    # return new_rhythms_and_expressions, polyphonic_bars
    
    
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

def add_barlines(rhythms_and_expressions, bar_numbers, offsets):
    """
    Adds barlines to input lists based on the bar numbers.
    """
    new_rhythms_and_expressions = []
    new_bar_numbers = []
    new_offsets = []
    curr_barline = None
    for i, (rhythm_and_expression, bar_number, offset) in enumerate(zip(rhythms_and_expressions, bar_numbers, offsets)):
        if bar_number != curr_barline:
            curr_barline = bar_number
            new_rhythms_and_expressions.append(BARLINE_TOKEN)
            new_bar_numbers.append(bar_number)
            new_offsets.append(offset)
        new_rhythms_and_expressions.append(rhythm_and_expression)
        new_bar_numbers.append(bar_number)
        new_offsets.append(offset)
    return new_rhythms_and_expressions, new_bar_numbers, new_offsets
    
def merge_notes(rhythms_and_expressions, offsets):
    """
    merges notes that have the same offset and length
    """
    new_rhythms_and_expressions = [rhythms_and_expressions[0]]
    # new_bar_lines = [bar_lines[0]]
    
    for i, (rhythm_and_expression, offset) in enumerate(zip(rhythms_and_expressions, offsets)):
        if i == 0:
            continue
        if offset == offsets[i-1] and rhythm_and_expression == rhythms_and_expressions[i-1]:
            continue
        new_rhythms_and_expressions.append(rhythms_and_expressions[i])
        # new_bar_lines.append(bar_lines[i-1])
    
    return new_rhythms_and_expressions


def get_base_tokenizer_dicts(want_barlines=False, no_expressions=True):
    """
    Creates base tokenizer dictionary without any actual notes
    """
    
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