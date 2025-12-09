# RNN language model pretrained on processed kern data
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RhythmLSTM(nn.Module):
    """
    LSTM-based language model to compute probability of score.
    """
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers): # defaulting to None?
        super(RhythmLSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)  # Predict next token
        self.device = torch.device("cpu")

    def forward(self, x, hidden=None, c=None, batched=False, lengths=None):
        """
        x: (batch_size, seq_length) or (seq_length) if batch_size is 1
        If batched, it should be padded
        hidden: want to pass in hidden state from previous forward pass
        lengths: lengths of sequences in the batch (if batched) for packing, tensor of shape (batch_size)
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = self.embedding(x)
                
        if batched:
            # print("here",lengths.shape, lengths.dtype, lengths.device
            if lengths is None or len(lengths) != x.shape[0]:
                raise ValueError("lengths must be provided for batched input")
            
            assert len(x.shape) > 1, "Input x must be at least 2D for batched processing"
            # lengths.to(self.device)
            
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            
        if hidden is None or c is None:
            out, (hidden, c) = self.lstm(x)
        else:
            out, (hidden, c) = self.lstm(x, (hidden, c)) # out: (batch_size, seq_length, hidden_size)
            
        if batched:
            out, _ = pad_packed_sequence(out, batch_first=True)

        out = self.fc(out) # shape: (batch_size, seq_length, vocab_size)
        if out.shape[0] == 1:
            out = out.squeeze(0)
        return out, (hidden, c)
    
    def to(self, device):
        super(RhythmLSTM, self).to(device)
        self.device = device
        
        return self
    def __str__(self):
        print(self.__dict__)