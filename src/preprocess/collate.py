import torch
from torch.utils.data import DataLoader, Dataset
import pickle as pkl
import json
import os
import numpy as np

from ..const_tokens import *
from ..utils import open_processed_data_dir

class TokenizedDataset(Dataset):
    """
    Used for formatting a tokenized dataset for pretraining
    """
    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): Dictionary with file names as keys and tokenized data as values.
        """
        self.part_id = [(piece_name, part) for piece_name in data_dict.keys() for part in data_dict[piece_name].keys() if part != "split"]
        self.tokenized_data = [torch.tensor(np.array(data_dict[piece_name][part], dtype=int), dtype=torch.long) for piece_name, part in self.part_id]

    def __len__(self):
        return len(self.part_id)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (file_name, tokenized_data)
        """
        return self.part_id[idx], self.tokenized_data[idx]

def collate_fn(batch, pad_token_id, device="cuda"):
    """
    collate
    """
    part_id = []
    data = []
    for b in batch:
        part_id.append(b[0])
        data.append(b[1])
    
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=pad_token_id).to(device)
    return part_id, data
    
def preprocess_data(data_path, batch_size, device="cuda") -> DataLoader:
    """
    Preprocess the data, data_path is a dictionary where the keys are the names of the files and the values are the tokenized data.
    """
    raw_data = json.load(open(os.path.join(data_path, "tokenized_dataset.json"), "r")) # TODO: update to use open_processed_data_dir
    token_to_id = pkl.load(open(os.path.join(data_path, "token_to_id.pkl"), "rb"))
    id_to_token = pkl.load(open(os.path.join(data_path, "id_to_token.pkl"), "rb"))
    # dataset = TokenizedDataset(data)
    pad_token_id = token_to_id[PADDING_TOKEN]
    
    train_dataset = TokenizedDataset({piece_name: raw_data[piece_name] for piece_name in raw_data.keys() if raw_data[piece_name]["split"] == "train"})
    val_dataset = TokenizedDataset({piece_name: raw_data[piece_name] for piece_name in raw_data.keys() if raw_data[piece_name]["split"] == "val"})
    
    # print(len(train_dataset), len(val_dataset))
    
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, pad_token_id, device))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, pad_token_id, device))
    
    return train_loader, val_loader, token_to_id, id_to_token
    

class Seq2SeqDataset(Dataset):
    """
    Dataset primarily to train the S2S barline transformer model.
    """
    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): Dictionary with file names as keys and tokenized data as values.
        """
        self.part_id = [(piece_name, part) for piece_name in data_dict.keys() for part in data_dict[piece_name].keys() if part != "split"]
        self.input_seq_tokenized_data = [torch.tensor(np.array(data_dict[piece_name][part]["no_barline"], dtype=int), dtype=torch.long) for piece_name, part in self.part_id]
        self.target_seq_tokenized_data = [torch.tensor(np.array(data_dict[piece_name][part]["barline"], dtype=int), dtype=torch.long) for piece_name, part in self.part_id]    

    def __len__(self):
        return len(self.part_id)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (file_name, input_seq_tokenized_data, target_seq_tokenized_data)
        """
        return self.part_id[idx], self.input_seq_tokenized_data[idx], self.target_seq_tokenized_data[idx]

def seq2seq_collate(batch, pad_token_id, device="cuda"):
    """
    collate for seq2seq dataset
    """
    part_id = []
    input_data = []
    target_data = []
    for b in batch:
        part_id.append(b[0])
        input_data.append(b[1])
        target_data.append(b[2])
    
    input_data = torch.nn.utils.rnn.pad_sequence(input_data, batch_first=True, padding_value=pad_token_id).to(device)
    target_data = torch.nn.utils.rnn.pad_sequence(target_data, batch_first=True, padding_value=pad_token_id).to(device)
    
    # create attention mark for input and target data
    src_key_padding_mask = (input_data == pad_token_id).to(device)
    tgt_key_padding_mask = (target_data == pad_token_id).to(device) 
        
    return part_id, input_data, target_data, src_key_padding_mask, tgt_key_padding_mask

def preprocess_parallel_data(data_dir, batch_size, device="cuda") -> DataLoader:
    """
    Preprocess the data, data_dir is a dictionary where the keys are the names of the files and the values are the tokenized data.
    """
    raw_data = json.load(open(os.path.join(data_dir, "parallel_barline_dataset.json"), "r"))
    token_to_id, id_to_token, _ = open_processed_data_dir(data_dir)
    
    pad_token_id = token_to_id[PADDING_TOKEN]
    
    train_dataset = Seq2SeqDataset({piece_name: raw_data[piece_name] for piece_name in raw_data.keys() if raw_data[piece_name]["split"] == "train"})
    val_dataset = Seq2SeqDataset({piece_name: raw_data[piece_name] for piece_name in raw_data.keys() if raw_data[piece_name]["split"] == "val"})
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: seq2seq_collate(x, pad_token_id, device))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: seq2seq_collate(x, pad_token_id, device))
    
    return train_loader, val_loader, token_to_id, id_to_token

if __name__ == "__main__":
    data_dir = "processed_data/all/parallel_barlines"
    batch_size = 4
    device = "cuda"
    train_loader, val_loader, token_to_id, id_to_token = preprocess_parallel_data(data_dir, batch_size, device)
    for batch in train_loader:
        part_id, input_data, target_data, input_mask, target_mask = batch
        print(part_id)
        print(input_data.shape)
        print(target_data.shape)
        print(input_mask.shape)
        print(target_mask.shape)
        break