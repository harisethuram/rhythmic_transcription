# rnn model pretrained on processed kern data
import torch
import torch.nn as nn

class RhythmLSTM(nn.Module):
    def __init__(self, vocab_size=None, embed_size=None, hidden_size=None, num_layers=None):
        super(RhythmLSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)  # Predict next token

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden) # out: (batch_size, seq_length, hidden_size)
        out = self.fc(out.reshape(-1, out.shape[2])) # shape: (batch_size * seq_length, vocab_size)
        return out, hidden