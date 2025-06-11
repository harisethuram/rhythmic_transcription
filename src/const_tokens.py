from fractions import Fraction

START_OF_SEQUENCE_TOKEN = "<SOS>"
END_OF_SEQUENCE_TOKEN = "<EOS>"
UNKNOWN_TOKEN = "<UNK>"
PADDING_TOKEN = "<PAD>"
BARLINE_TOKEN = "<BARLINE>"

QUARTER_LENGTHS = {Fraction(length/2): f"<QL_{Fraction(length/2)}>" for length in range(1, 17)}
# print(QUARTER_LENGTHS)
# input("Press enter to continue...")

CONST_TOKENS = set((START_OF_SEQUENCE_TOKEN, END_OF_SEQUENCE_TOKEN, UNKNOWN_TOKEN, PADDING_TOKEN, BARLINE_TOKEN)).union(QUARTER_LENGTHS.values())
# TODO: move this to src/constants.py