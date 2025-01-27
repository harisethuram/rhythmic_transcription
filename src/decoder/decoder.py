# takes in path to a note_info.json file, and a path to a pre-trained model.pth file, and outputs the joint decoded music output

import torch
import torch.nn as nn
import torch.nn.functional as F
import music21
from typing import List, Dict
import pickle as pkl
import os
from const_tokens import *

class Decoder(nn.Module):
    def __init__(self, language_model: nn.Module, channel_model: nn.Module, token_to_id: Dict, beam_width: int=5, temperature: float=1.0):
        super(Decoder, self).__init__()
        self.language_model = language_model
        self.channel_model = channel_model
        self.beam_width = beam_width
        self.temperature = temperature
        # with open(os.path.join(processed_data_dir, "id_to_token.pkl"), "rb") as f:
        #     self.id_to_token = pkl.load(f)
        # with open(os.path.join(processed_data_dir, "token_to_id.pkl"), "rb") as f:
        #     self.token_to_id = pkl.load(f)

    def decode(self, note_info: List, decode_method: str="greedy") -> List[int]:
        if decode_method == "greedy":
            return self._greedy_decode(note_info)
        elif decode_method == "sample":
            return self._sample_decode(note_info)
        elif decode_method == "beam_search":
            return self._beam_search(note_info)
        
        raise ValueError(f"Invalid decode method: {decode_method}, must be one of ['greedy', 'sample', 'beam_search']") 
            
    
    def _beam_search(note_info: List):
        pass
    
    def _greedy_decode(note_info: List):
        output = [START_OF_SEQUENCE_TOKEN]
        with open(note_info_path, "r") as f:
            note_info = json.load(f)
        
        output_id = [self.token_to_id[START_OF_SEQUENCE_TOKEN]]
        for i in range(len(note_info)):
            # for now, just predict the next token using output and language model
            logits, _ = self.language_model(torch.tensor(output_id).unsqueeze(0))
            next_token_logits = logits[:, -1, :].squeeze(0)
            
            # get the token with the highest probability
            next_token_id = torch.argmax(next_token_logits).item()
            
            # append the token to the output
            output_id.append(next_token_id)
            
            # check if the token is the end of sequence token
            if self.id_to_token[next_token_id] == END_OF_SEQUENCE_TOKEN:
                break
        print(output_id)
        return output_id
            
            
        
    def _sample_decode(note_info: List, temperature: float):
        pass
    
    