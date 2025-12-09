import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import music21
from typing import List, Dict, Tuple
import pickle as pkl
import os
from tqdm import tqdm
import numpy as np
from fractions import Fraction
import json
import sys

from ..model.GaussianChannel import GaussianChannel
from ..model.RhythmLSTM import RhythmLSTM
from ..utils import get_note_and_length_to_token_id_dicts, decompose_note_sequence, get_tokens_given_length, open_processed_data_dir
from ..note.Note import Note
from ..const_tokens import *

class GaussianDecoder(nn.Module):
    def __init__(self, rhythm_lstm: nn.Module, processed_data_dir: str, beam_width: int=5, lambda_param: float = 0.5, sigma: float = 0.1):
        """
        rhythm_lstm_path: path to the pre-trained rhythm LSTM model
        lambda_param: weight for combining the rhythm LSTM probabilities and Gaussian channel probabilities
        sigma: standard deviation for the Gaussian channel
        """
        super(GaussianDecoder, self).__init__()
        self.rhythm_lstm = rhythm_lstm
        self.rhythm_lstm.eval()
        
        self.gaussian_channel = GaussianChannel(sigma=sigma)
        self.lambda_param = lambda_param
        self.beam_width = beam_width
        self.token_to_id, self.id_to_token, _ = open_processed_data_dir(processed_data_dir)
        
    def forward(self, input_durations: List[float], is_note: List[bool], tempo: float, pitches: List[float]) -> Tuple[List[List[Note]], List[float], List[List[float]]]:
        """
        input_duration: list of note lengths
        is_note: list of booleans indicating whether the corresponding input_duration is a note (True) or a rest (False)
        tempo: tempo in BPM
        pitches: list of pitches corresponding to the input durations
        for now, we aren't dealing with staccato
        return: 
            best_sequences: List of best decoded note sequences (List of Notes)
            best_scores: List of scores for the best decoded sequences
        """
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        def to_tensor(lst, dtype=torch.long):
            return torch.tensor(lst, dtype=dtype).to(DEVICE)

        self.rhythm_lstm.to(DEVICE)
        self.gaussian_channel.to(DEVICE)
        
        num_lstm_layers = self.rhythm_lstm.lstm.num_layers
        
        # 1) construct set of all possible notes and rests
        print("Constructing possible notes and rests...")
        all_notes = [[event] for event in self.token_to_id.keys() if type(event) == Note and not event.is_rest and not event.tied_forward]
        all_rests = [[event] for event in self.token_to_id.keys() if type(event) == Note and event.is_rest]
        # print(len(all_notes))
        # however, we also need to account for ties, and consecutive rests. 
        # If there are t tied notes and t' untied notes, we need an additional t*t' possible tokens for tied notes in our vocab
        # for rests, if there are r rests, we need an additional r^2 rests in our vocab 
        tied_notes = []
        double_rests = []
        for note1 in self.token_to_id.keys():
            if note1 not in CONST_TOKENS and not note1.is_rest and note1.tied_forward:
                for note2 in self.token_to_id.keys():
                    if note2 not in CONST_TOKENS and not note2.is_rest and not note2.tied_forward:
                        tied_notes.append([note1, note2])
                    

        for rest1 in self.token_to_id.keys():
            for rest2 in self.token_to_id.keys():
                if rest1 not in CONST_TOKENS and rest2 not in CONST_TOKENS and rest1.is_rest and rest2.is_rest:
                    double_rests.append([rest1, rest2])
        
        all_notes += tied_notes
        all_rests += double_rests        

        all_notes = {i: notes for i, notes in enumerate(all_notes)} # mapping from gaussian id to list of Note events (notes)
        all_rests = {i: rests for i, rests in enumerate(all_rests)} # mapping from gaussian id to list of Note events (rests)
        
        # add a '0' rest token for very short durations
        zero_rest = Note(duration=0, dotted=False, triplet=False, fermata=False, staccato=False, tied_forward=False, is_rest=True)
        all_rests[len(all_rests)] = [zero_rest]
        # print(all_rests)
        # input()

        note_gaussian_scores = torch.zeros((len(input_durations), len(all_notes))).to(DEVICE) # (num input durations, num note tokens)
        rest_gaussian_scores = torch.zeros((len(input_durations), len(all_rests))).to(DEVICE) # (num input durations, num rest tokens)
        
        
        
        # 2) get all the gaussian channel probabilities
        print("Pre-Computing Gaussian channel probabilities...")
        for i in range(len(input_durations)):
            input_duration_tensor = to_tensor([input_durations[i]], dtype=torch.float)
            this_all = all_notes if is_note[i] else all_rests
            this_gaussian_scores = note_gaussian_scores if is_note[i] else rest_gaussian_scores
            # print(f"{i}:", is_note[i])
            
            for j, token_events in this_all.items():
                symbollic_duration = sum([event.get_len() for event in token_events])
                prob = self.gaussian_channel(input_duration_tensor, float(symbollic_duration), tempo)
                this_gaussian_scores[i, j] = prob.item()
        print("Gauss:", note_gaussian_scores.shape, rest_gaussian_scores.shape)
        
        # print("running tests...")
        # # i want a dict where the key is a str concat of note id and note.get_len(), and the value is the gaussian score for the first note
        # note_id_len_to_gaussian_score = {}
        # for j, token_events in all_notes.items():
        #     # note = token_events[0]
        #     key = f"{j}_{float(sum([note.get_len() for note in token_events]))}"
        #     note_id_len_to_gaussian_score[key] = note_gaussian_scores[0, j].item()
            
        # # sort dict by value
        # note_id_len_to_gaussian_score = dict(sorted(note_id_len_to_gaussian_score.items(), key=lambda item: item[1]))
        # os.makedirs("test/new/gaussian_tests", exist_ok=True)
        # with open(f"test/new/gaussian_tests/scores_{tempo}.json", "w") as f:
        #     json.dump(note_id_len_to_gaussian_score, f, indent=4)
        
        # # do the same for hte second note
        # rest_id_len_to_gaussian_score = {}
        # for j, token_events in all_rests.items():
        #     rest = token_events[0]
        #     key = f"{j}_{float(sum([event.get_len() for event in token_events]))}"
        #     rest_id_len_to_gaussian_score[key] = rest_gaussian_scores[1, j].item()
        # # sort dict by value
        # rest_id_len_to_gaussian_score = dict(sorted(rest_id_len_to_gaussian_score.items(), key=lambda item: item[1]))
        # with open(f"test/new/gaussian_tests/rest_scores_{tempo}.json", "w") as f:
        #     json.dump(rest_id_len_to_gaussian_score, f, indent=4)
        # input("complete tests")
         
        # 3) Beam search decoding
        prefixes = [to_tensor([self.token_to_id[START_OF_SEQUENCE_TOKEN]], dtype=torch.long)]  # initial prefix with START token
        best_pitches = [[None]]
        prefix_scores = [0.0]  # initial scores
        _, (tmp1, tmp2) = self.rhythm_lstm(prefixes[0])
        # print(first_token_logits.shape)
        # input()
        lstm_hidden_states = [tmp1]
        lstm_cell_states = [tmp2]
        
        with torch.no_grad():
            for i, (duration, is_n, pitch) in enumerate(tqdm(zip(input_durations, is_note, pitches))): # iterate over input sequence
                all_candidates = []
                all_pitches = []
                all_scores = []
                all_lstm_h = []
                all_lstm_c = []
                
                possible_tokens = all_notes if is_n else all_rests # this is a mapping to gaussian id, not token id 
                gaussian_scores = note_gaussian_scores[i, :] if is_n else rest_gaussian_scores[i, :] # get either note or rest gaussian scores
                
                # # parallelize the below loop now
                all_tmp = [(k, v) for k, v in possible_tokens.items() if v[0] != zero_rest]
                
                all_gaussian_ids = [k for k, _ in all_tmp]
                
                all_events = [v for _, v in all_tmp]
                
                # # # use torch.gather to get all gaussian scores from all_gaussian_ids
                curr_gaussian_scores = gaussian_scores[all_gaussian_ids].tolist() # (num possible events)
                all_tokenized_events = [to_tensor([self.token_to_id[event] for event in token_events]) for token_events in all_events]                
               
                num_tokenized_events = len(all_tokenized_events)
                num_beams = len(prefixes)
                h_0_batch = torch.cat(lstm_hidden_states, dim=1).repeat(1, num_tokenized_events, 1)  # (num_layers, num_beams * num_tokenized_events, hidden_size), repeated at batch level i.e. all beams one copy, then all beams again
                c_0_batch = torch.cat(lstm_cell_states, dim=1).repeat(1, num_tokenized_events, 1)  # (num_layers, num_beams * num_tokenized_events, cell_size)
                first_token_logits = self.rhythm_lstm.fc(h_0_batch[-1]) # (num_beams * num_tokenized_events, vocab_size)

                all_tokenized_events_repeated = [] # repeated by number of prefixes, at the token event level i.e. each token is repeated num_beams times before next token
                for token_event in all_tokenized_events:
                    all_tokenized_events_repeated.extend([token_event for _ in range(num_beams)])
                    
                curr_gaussian_scores_repeated = []
                for score in curr_gaussian_scores:
                    curr_gaussian_scores_repeated.extend([score for _ in range(num_beams)])
                    
                curr_gaussian_scores_repeated = to_tensor(curr_gaussian_scores_repeated, dtype=torch.float) # (num_beams * num_tokenized_events)

                padded_all_tokenized_events = pad_sequence(all_tokenized_events_repeated, batch_first=True, padding_value=self.token_to_id[PADDING_TOKEN]) # (num_beams * all_tokenized_events_repeated, max_seq_length)
                lengths = torch.Tensor([len(seq) for seq in all_tokenized_events_repeated]) # (num_possible_events)
                
                lstm_out_batch, (new_h_batch_tmp, new_c_batch_tmp) = self.rhythm_lstm(
                    padded_all_tokenized_events,
                    hidden=h_0_batch,
                    c=c_0_batch,
                    batched=True,
                    lengths=lengths
                ) # (num_beams * num_possible_events, seq_length, vocab_size), (num_layers, num_beams * num_possible_events, hidden_size), (num_layers, num_beams * num_possible_events, hidden_size)
                
                # now, for each sequence in the batch, compute the lm log probs and combine with gaussian log probs, batched
                lstm_out_batch = torch.cat((first_token_logits.unsqueeze(1), lstm_out_batch), dim=1)[:, :-1, :]  # prepend first token logits and remove last time step to align with input
                # lstm_out_batch, first_token_logits = lstm_out_batch[:, :-1, :], lstm_out_batch[:, -1, :]
                
                
                
                all_log_probs = F.log_softmax(lstm_out_batch, dim=-1)
                
                final_lm_log_probs = torch.gather(all_log_probs, -1, padded_all_tokenized_events.unsqueeze(-1)).squeeze(-1) # (num_beams * num_possible_events, seq_length)
                
                mask = (padded_all_tokenized_events != self.token_to_id[PADDING_TOKEN]).float() # (num_beams * num_possible_events, seq_length)
                
                assert torch.all((mask == 0) == (padded_all_tokenized_events == self.token_to_id[PADDING_TOKEN]))
                
                summed_lm_log_probs = (final_lm_log_probs * mask).sum(dim=1) # (num_beams * num_possible_events)
                gaussian_log_probs = torch.log(curr_gaussian_scores_repeated + 1e-10) # (num_beams * num_possible_events)
                combined_log_probs = (1 - self.lambda_param) * summed_lm_log_probs + self.lambda_param * gaussian_log_probs # (num_beams * num_possible_events)
                
                # repeat scores and prefixes accordingly
                prefixes_repeated = prefixes * num_tokenized_events # each prefix repeated num_possible_events times
                scores_repeated = prefix_scores * num_tokenized_events
                pitches_repeated = best_pitches * num_tokenized_events
                
                # for _ in range(num_tokenized_events):
                #     prefixes_repeated.extend(prefixes)
                #     scores_repeated.extend(prefix_scores)
                #     pitches_repeated.extend(best_pitches)
                
                # print(len(pitches_repeated))
                # input()
                scores_repeated = to_tensor(scores_repeated, dtype=torch.float) # (num_beams * num_possible_events)
                
                total_log_probs = scores_repeated + combined_log_probs  # add previous score to each (num_beams * num_possible_events)
                
                
                all_candidates = [torch.cat((prefixes_repeated[i], all_tokenized_events_repeated[i])) for i in range(len(all_tokenized_events_repeated))]
                all_scores = total_log_probs.tolist()
                all_pitches = [pitches_repeated[i] + [pitch] * int(lengths[i].item()) for i, length in enumerate(lengths)]
                all_lstm_h = [new_h_batch_tmp[:, i, :].unsqueeze(1) for i in range(new_h_batch_tmp.shape[1])]
                all_lstm_c = [new_c_batch_tmp[:, i, :].unsqueeze(1) for i in range(new_c_batch_tmp.shape[1])]
                
                # now also consider the zero rest token (no new token added)
                if not is_n:
                    total_log_probs = to_tensor(prefix_scores, dtype=torch.float) + torch.log(gaussian_scores[len(possible_tokens)-1] + 1e-10)  # (num_beams)
                    all_candidates.extend(prefixes)
                    all_scores.extend(total_log_probs.tolist())
                    all_lstm_h.extend(lstm_hidden_states)
                    all_lstm_c.extend(lstm_cell_states)
                    all_pitches.extend(best_pitches)
                    
                
                    
                                
                # for prefix, score, lstm_h, lstm_c, curr_pitches in zip(prefixes, prefix_scores, lstm_hidden_states, lstm_cell_states, best_pitches): # iterate over beam prefixes
                #     all_tmp = [(k, v) for k, v in possible_tokens.items() if v[0] != zero_rest]
                    
                #     all_gaussian_ids = [k for k, _ in all_tmp]
                    
                #     assert all([all_gaussian_ids[i] == i for i in range(len(all_gaussian_ids))]), "Gaussian IDs are not sequential!"
                #     all_events = [v for _, v in all_tmp]
                    
                #     # use torch.gather to get all gaussian scores from all_gaussian_ids
                #     curr_gaussian_scores = gaussian_scores[all_gaussian_ids] # (num possible events)
                #     all_tokenized_events = [to_tensor([self.token_to_id[event] for event in token_events]) for token_events in all_events]
                #     h_0_batch = lstm_h.repeat(1, len(all_tokenized_events), 1)  # (num_layers, num_possible_events, hidden_size)
                #     c_0_batch = lstm_c.repeat(1, len(all_tokenized_events), 1)  # (num_layers, num_possible_events, hidden_size)
                    
                #     # pad sequences
                #     padded_all_tokenized_events = pad_sequence(all_tokenized_events, batch_first=True, padding_value=self.token_to_id[PADDING_TOKEN]) # (num_possible_events, max_seq_length)
                #     lengths = torch.Tensor([len(seq) for seq in all_tokenized_events]) # (num_possible_events)
                    
                #     # forward pass through rhythm LSTM
                #     lstm_out_batch, (new_h_batch, new_c_batch) = self.rhythm_lstm(
                #         padded_all_tokenized_events, 
                #         hidden=h_0_batch, 
                #         c=c_0_batch, 
                #         batched=True, 
                #         lengths=lengths
                #     ) # (num_possible_events, seq_length, vocab_size), (num_layers, num_possible_events, hidden_size), (num_layers, num_possible_events, hidden_size)
                    
                #     # now, for each sequence in the batch, compute the lm log probs and combine with gaussian log probs, batched
                #     all_log_probs = F.log_softmax(lstm_out_batch, dim=-1) # (num_possible_events, seq_length, vocab_size)
                    
                #     # gather lm log probs for each token event
                #     final_lm_log_probs = torch.gather(all_log_probs, -1, padded_all_tokenized_events.unsqueeze(-1)).squeeze(-1) # (num_possible_events, seq_length)
                #     mask = (padded_all_tokenized_events != self.token_to_id[PADDING_TOKEN]).float() # (num_possible_events, seq_length)
                    
                #     assert torch.all((mask == 0) == (padded_all_tokenized_events == self.token_to_id[PADDING_TOKEN]))
                    
                #     summed_lm_log_probs = (final_lm_log_probs * mask).sum(dim=1)  # sum over sequence length, (num_possible_events)
                #     gaussian_log_probs = torch.log(curr_gaussian_scores + 1e-10) # (num_possible_events)
                #     combined_log_probs = (1 - self.lambda_param) * summed_lm_log_probs + self.lambda_param * gaussian_log_probs # (num_possible_events)
                #     total_log_probs = score + combined_log_probs  # add previous score to each (num_possible_events)
                    
                    
                #     # TODO: update all_candidates, all_scores, all_lstm_h, all_lstm_c, all_pitches with batched results
                #     scaled_prefixes = prefix.unsqueeze(0).repeat(len(all_tokenized_events), 1)  # (num_possible_events, seq_length)
                #     all_candidates += [torch.cat((scaled_prefixes[i], all_tokenized_events[i])) for i in range(len(all_tokenized_events))]
                #     all_scores += total_log_probs.tolist()
                #     all_pitches += [curr_pitches + [pitch] * int(length) for length in lengths.tolist()]
                #     all_lstm_h += [new_h_batch[:, i, :].unsqueeze(1) for i in range(new_h_batch.shape[1])]
                #     all_lstm_c += [new_c_batch[:, i, :].unsqueeze(1) for i in range(new_c_batch.shape[1])]
                    
                #     if not is_n: # also consider the zero rest token (no new token added)
                #         total_log_prob = score + torch.log(gaussian_scores[len(possible_tokens)-1] + 1e-10).item()
                        
                #         all_candidates.append(prefix) # no new token added
                #         all_scores.append(total_log_prob)
                #         all_lstm_h.append(lstm_h)
                #         all_lstm_c.append(lstm_c)
                #         all_pitches.append(curr_pitches)
                        
                if all_candidates:
                    top_k_indices = torch.topk(torch.tensor(all_scores), k=min(self.beam_width, len(all_scores))).indices
                    prefixes = [all_candidates[i] for i in top_k_indices]
                    prefix_scores = [all_scores[i] for i in top_k_indices]
                    lstm_hidden_states = [all_lstm_h[i] for i in top_k_indices]
                    lstm_cell_states = [all_lstm_c[i] for i in top_k_indices]
                    best_pitches = [all_pitches[i] for i in top_k_indices]
                    
        ### OUTDATED NOT PARALLELIZED CODE BELOW ###           
                    # tmp_compile = [(x, y) for x, y in zip(all_candidates, all_scores)]
                    # for x, y in tmp_compile[:40]:
                    #     print(x, y)
                    # print("...")
                    # print(tmp_compile[-5:])
                    # input()
                    
                    # if not is_n:
                #     argmax_1 = np.argmax(all_scores)
                #         # print(all_scores, argmax_1)

                #     # all_scores =  
                #     # not parallelized version of the above loop
                #     not_parallel_all_scores = []
                    
                #     for gaussian_id, token_events in possible_tokens.items():
                #         if token_events[0] == zero_rest: # we just want the gaussian prob, no lm `prob as no new token is added
                #             # print(gaussian_id, "gauss")
                #             gaussian_log_prob = torch.log(gaussian_scores[gaussian_id] + 1e-10).item()
                #             combined_log_prob = gaussian_log_prob
                #             curr_candidate = to_tensor(prefix) # no new token added
                #             new_lstm_h, new_lstm_c = lstm_h, lstm_c
                #             curr_candidate_pitches = curr_pitches
                #         else:
                #             tokenized = to_tensor([self.token_to_id[event] for event in token_events])
                #             # curr_candidate = to_tensor(prefix + tokenized)
                            
                #             curr_candidate_pitches = curr_pitches + [pitch] * len(token_events)
                #             curr_candidate = torch.cat((prefix, tokenized), dim=0)
                            
                #             lstm_out, (new_lstm_h, new_lstm_c) = self.rhythm_lstm(tokenized, lstm_h, lstm_c)
                            
                #             if len(token_events) == 1:
                #                 step1 = F.log_softmax(lstm_out[-1, :], dim=-1)
                #                 lm_log_probs = step1[self.token_to_id[token_events[0]]].item()
                #             elif len(token_events) == 2:
                #                 token1 = F.log_softmax(lstm_out[-2, :], dim=-1)
                #                 token2 = F.log_softmax(lstm_out[-1, :], dim=-1)
                #                 lm_log_probs = token1[self.token_to_id[token_events[0]]].item() + token2[self.token_to_id[token_events[1]]].item()
                #             else:
                #                 raise NotImplementedError("Tied notes more than length 2 not supported yet")
                                                        
                #             gaussian_log_prob = torch.log(gaussian_scores[gaussian_id] + 1e-10).item()
                #             combined_log_prob = (1 - self.lambda_param) * lm_log_probs + self.lambda_param * gaussian_log_prob

                #         # all_candidates.append(curr_candidate)
                #         not_parallel_all_scores.append(score + combined_log_prob)
                #         # all_lstm_h.append(new_lstm_h)
                #         # all_lstm_c.append(new_lstm_c)
                #         # all_pitches.append(curr_candidate_pitches)

                # # Select the top-k candidates
                # # if not is_n:
                # argmax_2 = np.argmax(not_parallel_all_scores)
                
                # if argmax_1 != argmax_2:
                #     print(not_parallel_all_scores, argmax_2)
                #     print("Mismatch between parallel and non-parallel beam search!")
                #     input()
                # if all_candidates:
                #     top_k_indices = torch.topk(torch.tensor(all_scores), k=min(self.beam_width, len(all_scores))).indices
                #     prefixes = [all_candidates[i] for i in top_k_indices]
                #     prefix_scores = [all_scores[i] for i in top_k_indices]
                #     lstm_hidden_states = [all_lstm_h[i] for i in top_k_indices]
                #     lstm_cell_states = [all_lstm_c[i] for i in top_k_indices]
                #     best_pitches = [all_pitches[i] for i in top_k_indices]
        # print("final prefixes:")
        # print(prefixes)
        ### END NON-PARALLELIZED CODE ###
        
        # add END token to each prefix and compute final scores
        best_sequences = []
        best_scores = []
        for prefix, score, lstm_h, lstm_c in zip(prefixes, prefix_scores, lstm_hidden_states, lstm_cell_states):
            curr_candidate = to_tensor(torch.cat((prefix, to_tensor([self.token_to_id[END_OF_SEQUENCE_TOKEN]]))))
            # lm_log_prob = F.log_softmax(lstm_out[-1, :], dim=-1)[self.token_to_id[END_OF_SEQUENCE_TOKEN]].item()
            lm_log_prob = self.rhythm_lstm.fc(lstm_h[-1])  # (1, vocab_size)
            lm_log_prob = F.log_softmax(lm_log_prob, dim=-1)[0, self.token_to_id[END_OF_SEQUENCE_TOKEN]].item()
            
            combined_log_prob = score + lm_log_prob
            best_sequences.append(curr_candidate)
            best_scores.append(combined_log_prob)
        # print("tok")
        # print(best_sequences)
        # convert best_sequences from token ids to Note sequences
        untokenized_best_sequences = []
        for seq in best_sequences:
            note_sequence = []
            for token_id in seq.cpu().numpy():
                token = self.id_to_token[token_id]
                note_sequence.append(token)
            untokenized_best_sequences.append(note_sequence)
        for best_pitch in best_pitches:
            best_pitch.append(None) # for the EOS token
        # print("unt")
        # print(untokenized_best_sequences)
        # print("length:", len(untokenized_best_sequences[0]), len(best_pitches[0]))
        # print(untokenized_best_sequences)
        return untokenized_best_sequences, best_scores, best_pitches
