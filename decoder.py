# takes in path to a note_info.json file, and a path to a pre-trained model.pth file, and outputs the joint decoded music output

import torch
import torch.nn as nn
import torch.nn.functional as F
import music21
from typing import List
from const_tokens import *

class Decoder(nn.Module):
    def __init__(self, model: nn.Module, beam_width: int=5, temperature: float=1.0):
        super(Decoder, self).__init__()
        self.model = model
        self.beam_width = beam_width
        self.temperature = temperature

    def decode(self, note_info_path: str, decode_method: str="greedy") -> List[music21.note.Note]:
        if decode_method == "greedy":
            return self._greedy_decode(note_info_path)
        elif decode_method == "sample":
            return self._sample_decode(note_info_path)
        elif decode_method == "beam_search":
            return self._beam_search(note_info_path)
        
        raise ValueError(f"Invalid decode method: {decode_method}, must be one of ['greedy', 'sample', 'beam_search']") 
            
    
    def _beam_search(note_info_path: str):
        
    
    def _greedy_decode(note_info_path: str):
        
    
    def _sample_decode(note_info_path: str, temperature: float):
        pass
    
    