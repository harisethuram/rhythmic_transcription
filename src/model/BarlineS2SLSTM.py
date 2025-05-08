# sequence to sequence LSTM that takes adds barlines to a barline-less sequence
import torch
import torch.nn as nn

class BarlineS2SLSTM(nn.Module):
    """
    LSTM-based model to add barlines to a barline-less sequence. 
    Input is a sequence of tokens, and output is a sequence of tokens with barlines added.
    """
    
    def __init__(self, vocab_size, embed_size, num_layers, bias_init=None):
        super(BarlineS2SLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.device = torch.device("cpu")
        
        # we don't need to use the positional encoding in the LSTM
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=embed_size, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size=embed_size, hidden_size=embed_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(embed_size, vocab_size)
        
        
        if bias_init is not None:
            self.fc.bias.data = torch.tensor(bias_init, dtype=torch.float32)
        
    def forward(self, src, tgt, hidden=None, c=None):
        """
        src: (batch_size, src_seq_length)
        tgt: (batch_size, tgt_seq_length)
        """
        # src and tgt are expected to be of shape (batch_size, seq_length)
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        
        # encode the source sequence
        src_out, (src_hidden, src_c) = self.encoder(src)
        
        # decode the target sequence
        tgt_out, (tgt_hidden, tgt_c) = self.decoder(tgt, (src_hidden, src_c))
        
        logits = self.fc(tgt_out)
        
        return logits, (src_hidden, src_c), (tgt_hidden, tgt_c)
    
    def to(self, device):
        super(BarlineS2SLSTM, self).to(device)
        self.device = device
        self.embedding.to(device)
        self.encoder.to(device)
        self.decoder.to(device)
        self.fc.to(device)
        
        return self