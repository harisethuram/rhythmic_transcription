# rnn model pretrained on processed kern data
import torch
import torch.nn as nn

class RhythmLSTM(nn.Module):
    def __init__(self, vocab_size=None, embed_size=None, hidden_size=None, num_layers=None):
        super(RhythmLSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)  # Predict next token
        self.device = torch.device("cpu")

    def forward(self, x, hidden=None):
        """
        x: (batch_size, seq_length) or (seq_length) if batch_size is 1
        hidden: want to pass in hidden state from previous forward pass
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden) # out: (batch_size, seq_length, hidden_size)
        out = self.fc(out) # shape: (batch_size, seq_length, vocab_size)
        if out.shape[0] == 1:
            out = out.squeeze(0)
        return out, hidden
    
    def to(self, device):
        super(RhythmLSTM, self).to(device)
        self.device = device
        
        return self