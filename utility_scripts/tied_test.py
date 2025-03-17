from collections import defaultdict

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


# Example usage:
# ties = [None, 'start', None, 'stop', 'start', 'continue', None, 'stop']
# pitches = ['C4',  'G4',  'C4',  'G4',  'C4', 'C4',  'D4',  'C4' ]
pitches = ['C4', 'G4', 'C4', 'G4', 'C4', 'C4', 'C4', 'D4', 'C4']
ties = ['start', 'start', 'continue', 'stop', 'start', 'stop', 'start', None, 'stop']
# ties = ['start', None, 'stop', 'start', 'continue', 'stop']
# pitches = ['C4', 'G4', 'C4', 'C4', 'C4', 'C4']
result = tie_chains_from_lists(ties, pitches)
print(result)
# e.g. [[1, 3], [4, 5, 7]]
