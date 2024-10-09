import torch
from torch.utils.data import DataLoader, Dataset
import pickle as pkl
import os

class TokenizedDataset(Dataset):
    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): Dictionary with file names as keys and tokenized data as values.
        """
        self.part_id = list(data_dict.keys())
        self.tokenized_data = [torch.tensor(data_dict[k], dtype=torch.long) for k in self.part_id]

    def __len__(self):
        return len(self.part_id)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (tokenized_sequence, file_name)
        """
        return self.part_id[idx], self.tokenized_data[idx]

def collate_fn(batch, pad_token_id, device="cuda"):
    part_id = []
    data = []
    for b in batch:
        part_id.append(b[0])
        data.append(b[1])
    
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=pad_token_id).to(device)
    return part_id, data
    
def preprocess_data(data_path, batch_size, device="cuda") -> DataLoader:
    """
    Preprocess the data, data_path is a dictionary where the keys are the names of the files and the values are the tokenized data. Assumes pad_token_id is 0.
    """
    data = pkl.load(open(os.path.join(data_path, "tokenized_dataset.pkl"), "rb"))
    token_to_id = pkl.load(open(os.path.join(data_path, "token_to_id.pkl"), "rb"))
    id_to_token = pkl.load(open(os.path.join(data_path, "id_to_token.pkl"), "rb"))
    dataset = TokenizedDataset(data)
    pad_token_id = 0
    
    # randomly split the data into training and validation sets
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    torch.manual_seed(0)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, pad_token_id, device))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, pad_token_id, device))
    
    return train_loader, val_loader, token_to_id, id_to_token
    
    