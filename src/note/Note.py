from fractions import Fraction
import json

class Note:
    """
    Abstraction of note tuple, for easier attribute access
    """
    def __init__(self, duration=1, dotted=False, triplet=False, fermata=False, staccato=False, tied_forward=False, is_rest=False, string=None, tple=None, note=None):
        if note is not None:
            if not isinstance(note, Note):
                raise ValueError(f"Expected note to be of type Note, got {type(note)}")
            self.duration = note.duration
            self.dotted = note.dotted
            self.triplet = note.triplet
            self.fermata = note.fermata
            self.staccato = note.staccato
            self.tied_forward = note.tied_forward
            self.is_rest = note.is_rest
        elif string is not None: # in form of "Note(duration=_, dotted=_, triplet=_, fermata=_, staccato=_, tied_forward=_, is_rest=_)"
            string = string[6:-1]
            string = string.split(", ")
            def clean(s):
                return s.split("=")[1]
            string = [clean(s) for s in string]
            if len(string) != 7:
                raise ValueError(f"Invalid string: {string}")
            self.duration = Fraction(string[0])
            self.dotted = string[1] == "True"
            self.triplet = string[2] == "True"
            self.fermata = string[3] == "True"
            self.staccato = string[4] == "True"
            self.tied_forward = string[5] == "True"
            self.is_rest = string[6] == "True"
        elif tple is not None:
            self.duration = tple[0]
            self.dotted = Fraction(tple[1])
            self.triplet = tple[2]
            self.fermata = tple[3]
            self.staccato = tple[4]
            self.tied_forward = tple[5]
            self.is_rest = tple[6]
        else:
            self.duration = Fraction(duration)
            self.dotted = dotted
            self.triplet = triplet
            self.fermata = fermata
            self.staccato = staccato
            self.tied_forward = tied_forward
            self.is_rest = is_rest
        
        
        
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Note):
            return False
        return self.__dict__ == other.__dict__
        
    def compare_without_expressions(self, other):
        return self.duration == other.duration and self.dotted == other.dotted and self.triplet == other.triplet and self.is_rest == other.is_rest and self.tied_forward == other.tied_forward

    def get_len(self):
        return Fraction(self.duration) * (Fraction(3, 2) if self.dotted else 1) * (Fraction(2, 3) if self.triplet else 1)
        
    def __str__(self):
        return f"Note(duration={self.duration}, dotted={self.dotted}, triplet={self.triplet}, fermata={self.fermata}, staccato={self.staccato}, tied_forward={self.tied_forward}, is_rest={self.is_rest})" 

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash((self.duration, self.dotted, self.triplet, self.fermata, self.staccato, self.tied_forward, self.is_rest))
    
if __name__ == "__main__":
    print("Testing Note class...")
    note1 = Note(1, dotted=True, triplet=False, fermata=False, staccato=False, tied_forward=False, is_rest=False)
    print(note1)
    note2 = Note(1, dotted=True, triplet=False, fermata=False, staccato=False, tied_forward=False, is_rest=False)
    print([note1, note2])
    print(note1 == note2)
    print(note1.get_len())
    s = set([note1, note2])
    print(s)
    note3 = Note(string=str(note1))
    print(note3)
    print(note3 == note1)
    
    # save to json
    note_info = [note1, note2]
    with open("note_info.json", "w") as f:
        json.dump(note_info, f, default=lambda x: x.__str__())