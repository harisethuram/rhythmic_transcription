# takes in path to a note_info.json file, and a path to a pre-trained model.pth file, and outputs the joint decoded music output

import torch
import torch.nn as nn
import torch.nn.functional as F
import music21
from typing import List, Dict
import pickle as pkl
import os

from ..model.BetaChannel import BetaChannel
from..model.RhythmLSTM import RhythmLSTM
from const_tokens import *

class Decoder(nn.Module):
    def __init__(self, language_model: RhythmLSTM, beta_channel_model: BetaChannel, token_to_id: Dict, beam_width: int=5, temperature: float=1.0):
        super(Decoder, self).__init__()
        self.language_model = language_model
        self.beta_channel_model = beta_channel_model
        self.token_to_id = token_to_id
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
            
    
    def _beam_search(self, note_info: List):
        pass
    
    def _greedy_decode(self, note_info: List):
        # with open(note_info_path, "r") as f:
        #     note_info = json.load(f)
        output = torch.tensor([self.token_to_id[START_OF_SEQUENCE_TOKEN]])
        output = output.to(self.language_model.device)
        
        note_lengths = torch.Tensor([float(note[0]) for note in note_info])
        note_portions = torch.Tensor([float(note[1]) for note in note_info]) # (num_notes)
        
        beta_probs = self.beta_channel_model(note_portions) # (num_notes, num_dists)
        
        # for each beta_prob, we need to multiply
            
        for i in range(len(note_info)):
            # for now, just predict the next token using output and language model
            logits, _ = self.language_model(output)
            next_token_logits = logits[..., -1, :].unsqueeze(0)
            
            # get the token with the highest probability
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            
            # append the token to the output
            # print(output.device, next_token_id.device)
            # print(output.shape, next_token_id.shape, next_token_id, next_token_logits.shape)
            output = torch.cat([output, next_token_id])
            
            # check if the token is the end of sequence token
            if next_token_id.item() == self.token_to_id[END_OF_SEQUENCE_TOKEN]:
                break
            
        print(output)
        return output.tolist()
            
            
        
    def _sample_decode(self, note_info: List):
        pass
    
    