# sequence to sequence transformer that takes adds barlines to a barline-less sequence
import torch
import torch.nn as nn
from .PositionalEncoding import PositionalEncoding

class BarlineS2STransformer(nn.Module):
    """
    Transformer-based model to add barlines to a barline-less sequence. 
    Input is a sequence of tokens, and output is a sequence of tokens with barlines added.
    """
    
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, max_len=5000):
        super(BarlineS2STransformer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.positional_encoding = PositionalEncoding(d_model=embed_size, max_len=max_len)
        self.transformer = nn.Transformer(d_model=embed_size, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)
        
    def forward(self, src, tgt, tgt_mask=None):
        """
        src: (batch_size, src_seq_length)
        tgt: (batch_size, tgt_seq_length)
        """
        # src and tgt are expected to be of shape (batch_size, seq_length)
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = self.positional_encoding(src, batch_first=True)
        tgt = self.positional_encoding(tgt, batch_first=True)
        
        # pass through transformer
        tgt_is_causal = tgt_mask is not None
        out = self.transformer(src, tgt, tgt_mask=tgt_mask, tgt_is_causal=tgt_is_causal)
        out = self.fc(out)
        return out
        # out: (batch_size, tgt_seq_length, vocab_size)
    
    def to(self, device):
        super(BarlineS2STransformer, self).to(device)
        self.device = device
        
        return self