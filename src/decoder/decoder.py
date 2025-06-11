# takes in path to a note_info.json file, and a path to a pre-trained model.pth file, and outputs the joint decoded music output

import torch
import torch.nn as nn
import torch.nn.functional as F
import music21
from typing import List, Dict
import pickle as pkl
import os
from tqdm import tqdm
import numpy as np
from fractions import Fraction

from ..model.BetaChannel import BetaChannel
from ..model.RhythmLSTM import RhythmLSTM
from ..utils import get_note_and_length_to_token_id_dicts, decompose_note_sequence
from ..note.Note import Note
from const_tokens import *

class Decoder(nn.Module):
    """
    Class that does the decoding of the output of an onset quantizer, into a sequence of notes. 
    """
    def __init__(self, language_model: RhythmLSTM, beta_channel_model: BetaChannel, token_to_id: Dict, id_to_token: Dict, beam_width: int=5, base_value: float=1.0, unk_score=-1000):
        super(Decoder, self).__init__()
        self.language_model = language_model
        self.beta_channel_model = beta_channel_model
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.beam_width = beam_width
        self.base_value = Fraction.from_float(base_value)
        self.unk_score = unk_score

    def decode(self, note_info: List, decode_method: str="greedy", flatten: bool=True, want_all_candidates: bool=False, debug: bool=False) -> List[int]: # TODO: make sure to output detokenized also in case of flattened
        """
        Decodes the given note info using the specified decoding method.
        note_info: List of length 3 (or 4) arrays: [quantized_length, note_portion, rest_portion, pitch (optional)]
        decode_method: Decoding method to use, one of ['greedy', 'beam_search']
        flatten: If False, returns a list of lists, where each sublist corresponds to an onset. If True, returns a flat list of tokens.
        want_all_candidates: If True, returns all beam_width candidates of the beam search, otherwise returns only the best candidate.
        """
        if decode_method == "greedy":
            out = self._greedy_decode(note_info, debug=debug)
        elif decode_method == "beam_search":
            out = self._beam_search(note_info, want_all_candidates=want_all_candidates, debug=debug)
        else:
            raise ValueError(f"Invalid decode method: {decode_method}, must be one of ['greedy', 'beam_search']") 
        
        if flatten:
            return out
        # for testing
        all_result = [decompose_note_sequence(o, self.token_to_id, self.id_to_token) for o in out]
        result = [item[0] for item in all_result]
        detokenized_result = [item[1] for item in all_result]
        print("result:", result, len(result))
        if debug:
            print("Flat:", out)
            print("Not Flat:", result)
            input()
        return result, detokenized_result
            
    def _beam_search(self, note_info: List, want_all_candidates: bool=False, debug=False):
        def to_help(tensor):
            return tensor.to(self.language_model.device).to(torch.int64)
        
        def to_tensor(obj):
            x = to_help(torch.Tensor([obj]))
            if len(x.shape) == 2:
                x = x.squeeze()
            return x
        
        note_lengths = [float(note[0]) for note in note_info]
        note_portions = torch.Tensor([float(note[1]) for note in note_info])
        
        modes = np.array(self.beta_channel_model.modes)
        note_lengths_as_fraction = np.expand_dims(np.vectorize(Fraction.from_float)(note_lengths), -1) / self.base_value
        # print("notelength", note_lengths_as_fraction)
        # print("basevalue", note_lengths_as_fraction / self.base_value)
        candidate_notes_lengths = note_lengths_as_fraction * np.expand_dims(modes, 0)
        candidate_rests_lengths = note_lengths_as_fraction * np.expand_dims(1 - modes, 0)
        all_beta_probs = self.beta_channel_model(note_portions) # all (num_notes, num_dists)
        all_beta_probs = F.logsigmoid(all_beta_probs)
        
        note_length_to_id, rest_length_to_id = get_note_and_length_to_token_id_dicts(self.id_to_token)

        results = [[self.token_to_id[START_OF_SEQUENCE_TOKEN]] for _ in range(self.beam_width)]
        
        prefixes = [to_tensor(self.token_to_id[START_OF_SEQUENCE_TOKEN])]
        beam_log_probs = [0.0] # log probabilities of each beam upto this point
        
        tmp1, (tmp2, tmp3) = self.language_model(prefixes[0])     
        next_token_log_probs_all_beams = [F.log_softmax(tmp1, dim=-1)]
        hidden_states = [tmp2]
        cell_states = [tmp3]        

        print("all_beta_probs:", len(all_beta_probs))
        
        self.language_model.eval()
        
        with torch.no_grad():
            for i, (beta_probs, candidate_note_lengths, candidate_rest_lengths) in tqdm(enumerate(zip(all_beta_probs, candidate_notes_lengths, candidate_rests_lengths))):
                candidates_all_beams = [[] for _ in range(len(prefixes))] 
                beta_log_probs = beta_probs
                # for each element of prefixes, we want to compute the probabilities of all candidates. 
                # we want two arrays, first one consists of all the prefices of length beam_width, and the second consists of all tuples of (candidates, probability) for each prefix of shape (beam_width, num_candidates)
                # then we select the top beam_width candidates from second array, and choose the corresponding prefices from the first array and do the concatenation
                # sounds like a plan :)
                
                
                for beam_no, (prefix, hidden_state, cell_state, next_token_log_probs) in enumerate(zip(prefixes, hidden_states, cell_states, next_token_log_probs_all_beams)):
                    # print("beam_no:", beam_no)
                    for beta_log_prob, candidate_note_length, candidate_rest_length in zip(beta_log_probs, candidate_note_lengths, candidate_rest_lengths):
                        candidate_note_ids = note_length_to_id.get(candidate_note_length, [])
                        candidate_rest_ids = rest_length_to_id.get(candidate_rest_length, []) 
                        # if candidate note ids is empty, or candidate rest ids is empty and candidate rest length is not 0, then we can't proceed so we append an unk token
                        if len(candidate_note_ids) == 0 or (len(candidate_rest_ids) == 0 and candidate_rest_length != 0):
                            candidates_all_beams[beam_no].append((self.unk_score, to_tensor(self.token_to_id[UNKNOWN_TOKEN])))
                            continue
                        
                        for candidate_note_id in candidate_note_ids:
                            candidate_note_id = to_tensor(candidate_note_id)
                            if len(candidate_note_id) == 1:
                                note_prob = next_token_log_probs[..., -1, candidate_note_id]
                            else: # in this case we have a tied note
                                note1_prob = next_token_log_probs[..., -1, candidate_note_id[0]]
                                note2_prob, _ = self.language_model(to_tensor(candidate_note_id[0]), hidden_state, cell_state)
                                note2_prob = F.log_softmax(note2_prob, dim=-1)[..., -1, candidate_note_id[1]]
                                note_prob = note1_prob + note2_prob
                                
                            rest_logits, (note_hidden_state, note_cell_state) = self.language_model(candidate_note_id, hidden_state, cell_state)
                            rest_probs = F.log_softmax(rest_logits, dim=-1) 
                            
                            if candidate_rest_length == 0: # in this case there is only two choices: just the note, or the note followed by zero rest (this case is handled by the loop below)
                                candidates_all_beams[beam_no].append((note_prob + beta_log_prob, candidate_note_id))
                                
                            
                            for candidate_rest_id in candidate_rest_ids:        
                                candidate_rest_id = to_tensor(candidate_rest_id)
                                if len(candidate_rest_id) == 1:
                                    rest_prob = rest_probs[..., -1, candidate_rest_id]
                                else: # in this case we have two consecutive rests
                                    rest1_prob = rest_probs[..., -1, candidate_rest_id[0]]
                                    rest2_prob, _ = self.language_model(to_tensor(candidate_rest_id[0]), note_hidden_state)
                                    rest2_prob = F.log_softmax(rest2_prob, dim=-1)[..., -1, candidate_rest_id[1]]
                                    rest_prob = rest1_prob + rest2_prob
                                
                                next_prediction = torch.cat([candidate_note_id, candidate_rest_id], dim=-1)
                                prob = note_prob + rest_prob + beta_log_prob
                                
                                candidates_all_beams[beam_no].append((prob, next_prediction))
                                
                # now we have all the candidates for all the beams, we need to select the top beam_width candidates for each beam
                compiled_candidates = [] # list of tuples of (total_log_prob, beam_no, candidate)
                # print(len(beam_log_probs), len(candidates_all_beams))
                for beam_no, (beam_log_prob, candidates) in enumerate(zip(beam_log_probs, candidates_all_beams)):
                    for curr_prob, candidate in candidates:
                        # print("candidate:", candidate, curr_prob, beam_log_prob)
                        # input()
                        compiled_candidates.append((beam_log_prob + curr_prob, beam_no, candidate))
                        
                compiled_candidates = sorted(compiled_candidates, key=lambda x: x[0], reverse=True)
                top_candidates = compiled_candidates[:self.beam_width]
                
                new_prefixes = []
                new_beam_log_probs = []
                new_next_token_log_probs_all_beams = []
                new_hidden_states = []
                new_cell_states = []
                
                for total_log_prob, beam_no, candidate in top_candidates:
                    # print(beam_no)
                    new_prefixes.append(torch.cat([prefixes[beam_no], candidate], dim=-1))
                    new_beam_log_probs.append(total_log_prob)
                    new_next_token_logits, (hidden_state, cell_state) = self.language_model(candidate, hidden_states[beam_no], cell_states[beam_no])
                    new_next_token_log_probs_all_beams.append(F.log_softmax(new_next_token_logits, dim=-1))
                    new_hidden_states.append(hidden_state)
                    new_cell_states.append(cell_state)
                    
                prefixes = new_prefixes
                beam_log_probs = new_beam_log_probs
                next_token_log_probs_all_beams = new_next_token_log_probs_all_beams
                hidden_states = new_hidden_states
                cell_states = new_cell_states
        # return the best beam
        if want_all_candidates:
            ans = []
            for prefix, log_prob in zip(prefixes, beam_log_probs):
                ans.append(prefix.tolist())
        else:
            best_beam = np.argmax([i.item() for i in beam_log_probs])
            ans = [prefixes[best_beam].tolist()]
        return ans
    
    
    def _greedy_decode(self, note_info: List):
        # with open(note_info_path, "r") as f:
        #     note_info = json.load(f)
        def to_tensor(obj):
            x = torch.Tensor([obj]).to(self.language_model.device).to(torch.int64)
            if len(x.shape) == 2:
                x = x.squeeze()
            return x
        
        note_lengths = [float(note[0]) for note in note_info]
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
        # print("\n\n\n")
        modes = np.array(self.beta_channel_model.modes)
        # print(modes)
        note_lengths_as_fraction = np.expand_dims(np.vectorize(Fraction.from_float)(note_lengths), -1)
        # print("note_lengths as fraction:", note_lengths_as_fraction)
        candidate_notes_lengths = note_lengths_as_fraction * np.expand_dims(modes, 0)
        candidate_rests_lengths = note_lengths_as_fraction * np.expand_dims(1 - modes, 0)
        all_beta_probs = self.beta_channel_model(note_portions) # all (num_notes, num_dists)
        # apply a log sigmoid to the beta probs to ensure they are between 0 and 1 in log space
        all_beta_probs = F.logsigmoid(all_beta_probs)
        
        note_length_to_id, rest_length_to_id = get_note_and_length_to_token_id_dicts(self.id_to_token)
        
        start = torch.tensor([self.token_to_id[START_OF_SEQUENCE_TOKEN]])
        start = to_tensor(start)
        
        hidden_state = None
        cell_state = None
        result = [self.token_to_id[START_OF_SEQUENCE_TOKEN]]
        next_token_logits, (hidden_state, cell_state) = self.language_model(start)
        actual_lengths = note_lengths_as_fraction.squeeze().tolist()
        
        # print("\n\n")
        self.language_model.eval()
        with torch.no_grad():
            for i, (beta_probs, candidate_note_lengths, candidate_rest_lengths) in tqdm(enumerate(zip(all_beta_probs, candidate_notes_lengths, candidate_rests_lengths))):
                # next_token_logits, (hidden_state, c) = self.language_model(start, hidden_state, c)
                next_token_probs = F.log_softmax(next_token_logits, dim=-1)
                beta_log_probs = beta_probs
                # print(next_token_probs.shape, beta_log_probs.shape, candidate_note_lengths.shape, candidate_rest_lengths.shape)
                # print("candidate note lengths:", candidate_note_lengths, candidate_rest_lengths)
                # input()
                
                all_candidates = []
                
                for beta_log_prob, candidate_note_length, candidate_rest_length in zip(beta_log_probs, candidate_note_lengths, candidate_rest_lengths):
                    candidate_note_ids = note_length_to_id.get(candidate_note_length, [])
                    candidate_rest_ids = rest_length_to_id.get(candidate_rest_length, [])
                    
                    # print("candidate lengths:", candidate_note_ids, candidate_rest_length)
                    
                    
                    for candidate_note_id in candidate_note_ids:
                        # print("candidate note id raw:", candidate_note_id, to_tensor(candidate_note_id))
                        candidate_note_id = to_tensor(candidate_note_id)
                        if len(candidate_note_id) == 1:
                            note_prob = next_token_probs[..., -1, candidate_note_id]
                        else: # in this case we have a tied note
                            note1_prob = next_token_probs[..., -1, candidate_note_id[0]]
                            note2_prob, _ = self.language_model(to_tensor(candidate_note_id[0]), hidden_state, cell_state)
                            note2_prob = F.log_softmax(note2_prob, dim=-1)[..., -1, candidate_note_id[1]]
                            note_prob = note1_prob + note2_prob
                            
                            
                        
                        # if len(candidate_note_id.shape) == 2:
                        #     candidate_note_id = candidate_note_id.squeeze()
                            
                        # print("candidate note id tensor:", candidate_note_id)
                        rest_logits, (note_hidden_state, note_cell_state) = self.language_model(candidate_note_id, hidden_state, cell_state)
                        rest_probs = F.log_softmax(rest_logits, dim=-1) 
                        
                        if candidate_rest_length == 0: # in this case there is only two choices: just the note, or the note followed by zero rest (this case is handled by the loop below)
                            # print(note_prob, beta_log_prob, candidate_note_id)
                            all_candidates.append((note_prob + beta_log_prob, candidate_note_id))
                            
                        
                        for candidate_rest_id in candidate_rest_ids:
                            
                            # input()
                            candidate_rest_id = to_tensor(candidate_rest_id)
                            # if len(candidate_rest_id.shape) == 2:
                            #     candidate_rest_id = candidate_rest_id.squeeze()
                            # print(candidate_note_id, candidate_rest_id)    
                            if len(candidate_rest_id) == 1:
                                rest_prob = rest_probs[..., -1, candidate_rest_id]
                            else: # in this case we have two consecutive rests
                                rest1_prob = rest_probs[..., -1, candidate_rest_id[0]]
                                # print("candidate rest:", candidate_rest_id, to_tensor(candidate_rest_id[0]))
                                rest2_prob, _ = self.language_model(to_tensor(candidate_rest_id[0]), note_hidden_state)
                                rest2_prob = F.log_softmax(rest2_prob, dim=-1)[..., -1, candidate_rest_id[1]]
                                rest_prob = rest1_prob + rest2_prob
                            
                            next_prediction = torch.cat([candidate_note_id, candidate_rest_id], dim=-1)
                            prob = note_prob + rest_prob + beta_log_prob
                            # if prob > 0:
                            #     print("prob:", prob, note_prob, rest_prob, beta_log_prob)
                            #     print("candidate note id:", candidate_note_id)
                            #     print("candidate rest id:", candidate_rest_id)
                            # print("next prob:", (prob, next_prediction))
                            # input()
                            all_candidates.append((prob, next_prediction))
                            

                # print(all_candidates)
                # for candidate in all_candidates:
                #     print(candidate, len(candidate))
                
                # tmp = all_candidates[max(enumerate(all_candidates), key=lambda x: x[1][0])]
                best_next = None
                best_prob = float("-inf")
                for candidate in all_candidates:
                    if candidate[0] > best_prob:
                        best_next = candidate[1]
                        best_prob = candidate[0]
                # print("best next:", best_next, best_prob)        
                
                # start = torch.Tensor([next_note, next_rest]).to(self.language_model.device)
                
                next_token_logits, (hidden_state, cell_state) = self.language_model(best_next, hidden_state, cell_state)
                start = best_next[-1]
                result += best_next.tolist()
                # print("result:", result)
                # print("actual lenghths:", note_lengths_as_fraction[:i+1].tolist())
                # input()
                # _, hidden_state = self.language_model(start, hidden_state)
                # result.append(next_note)
                # if next_rest is not None:
                #     result.append(next_rest)
        # print("result:", result)        
        return result
        
    def _sample_decode(self, note_info: List):
        raise NotImplementedError("Sampling not yet implemented")
    

if __name__ == "__main__":
    result = [[[], []] for _ in range(len(note_info))]
    j = 0
    # out = [Note()]
    # for i in range(len(out)):
    #     if out[i] == self.token_to_id[START_OF_SEQUENCE_TOKEN]:
    #         continue
    #     # increment j if its a new onset: i is a note and i-1 is a rest, edge case: i > 1 as you have sos at start
    #     if not self.id_to_token[out[i]].is_rest and self.id_to_token[out[i-1]].is_rest and i > 1:
    #         j += 1 
    #     if not self.id_to_token[out[i]].is_rest:
    #         result[j][0].append(out[i])
    #     else:
    #         result[j][1].append(out[i])
    # return result