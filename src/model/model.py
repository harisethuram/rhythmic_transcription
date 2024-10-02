# rnn model pretrained on processed kern data
import torch
import torch.nn as nn

class RhythmLSTM(nn.Module):
    def __init__():
        super(RhythmLSTM, self).__init__()
        self.lstm = nn.LSTM(1, 1, 1)
        self.fc = nn.Linear(1, 1)