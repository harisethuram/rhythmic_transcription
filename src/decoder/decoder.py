# takes in path to a note_info.json file, and a path to a pre-trained model.pth file, and outputs the joint decoded music output

import torch
import torch.nn as nn
import torch.nn.functional as F
import music21
from typing import List, Dict
import pickle as pkl
import os
from tqdm import tqdm

from ..model.BetaChannel import BetaChannel
from ..model.RhythmLSTM import RhythmLSTM
from ..utils import get_note_and_length_to_token_id_dicts
from const_tokens import *

class Decoder(nn.Module):
    def __init__(self, language_model: RhythmLSTM, beta_channel_model: BetaChannel, token_to_id: Dict, id_to_token: Dict, beam_width: int=5, temperature: float=1.0):
        super(Decoder, self).__init__()
        self.language_model = language_model
        self.beta_channel_model = beta_channel_model
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
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
        start = torch.tensor([self.token_to_id[START_OF_SEQUENCE_TOKEN]])
        start = output.to(self.language_model.device)
        
        note_lengths = torch.Tensor([float(note[0]) for note in note_info])
        note_portions = torch.Tensor([float(note[1]) for note in note_info]) # (num_notes)
        
        
        # so we want to compute the joint probability of the next note. 
        # Currently, the language model outputs the probability distribution of the next note
        # We also have the actual note lengths. 
        # we can get the length of sound discretized using the beta channel model
        # so what we really want is, given the current note length and its sound portion, 
        #   compute the probability using each beta distribution to find all num_dists note/rest lengths. 
        #      this will be only around 5 or 6 choices
        #   then, for each of these lengths, compute the probability of that note length followed by the complimentary rest length using the language model
        #      in a little more depth, for each note length, index into the outputs distribution by doing a mapping of length to token_id to get all valid tokens
        #      for each valid token, get the probability and pass that through the lm to get the complimentary rest
        #      then, the overall probability for this note length for a beta choice and a token/rest choice is (beta_prob * token_prob) * rest_prob 
        #         note that we don't multiply the rest_prob by beta_prob as that is just 1 as it follows, and because the beta_prob encompasses both note and rest.
        
        # get all the candidate note lengths
        modes = torch.Tensor(self.beta_channel_model.modes)
        
        candidate_notes_lengths = note_lengths.unsqueeze(-1) * modes.unsqueeze(0)
        candidate_rests_lengths = note_lengths.unsqueeze(-1) * (1 - modes).unsqueeze(0)
        all_beta_probs = self.beta_channel_model(note_portions) # all (num_notes, num_dists)
                
        note_length_to_id, rest_length_to_id = get_note_and_length_to_token_id_dicts(self.id_to_token)
        hidden_state = None
        result = []
        
        for i, (beta_probs, canditate_note_lengths, candidate_rest_lengths) in tqdm(enumerate(zip(all_beta_probs, candidate_notes_lengths, candidate_rests_lengths))):
            next_token_logits, hidden_state = self.language_model(start, hidden_state)
            next_token_probs = F.log_softmax(next_token_logits, dim=-1)
            beta_log_probs = torch.log(beta_probs)
            
            all_candidates = []
            
            for beta_log_prob, candidate_note_length, candidate_rest_length in zip(beta_log_probs, canditate_note_lengths, candidate_rest_lengths):
                candidate_note_ids = note_length_to_id.get(candidate_note_length.item(), [])
                candidate_rest_ids = rest_length_to_id.get(candidate_rest_length.item(), [])
                
                for candidate_note_id in candidate_note_ids:
                    note_prob = next_token_probs[..., -1, candidate_note_id]
                    rest_logits, _ = self.language_model(torch.Tensor([candidate_rest_length_id]).to(self.language_model.device), hidden_state)
                    rest_probs = F.log_softmax(rest_logits, dim=-1)
                    for candidate_rest_id in candidate_rest_ids:
                        rest_prob = rest_probs[..., -1, candidate_rest_id]
                        all_candidates.append((note_prob + rest_prob + beta_log_prob, candidate_note_id, candidate_rest_id))
                        
            tmp = all_candidates[max(enumerate(all_candidates), key=lambda x: x[1][0])]
            next_note = tmp[1]
            next_rest = tmp[2]
            
            start = torch.Tensor([next_note, next_rest]).to(self.language_model.device)
            _, hidden_state = self.language_model(start, hidden_state)
            result.append(next_note)
            if next_rest is not None:
                result.append(next_rest)
                
        return result
            
            
        
    def _sample_decode(self, note_info: List):
        pass
    
    