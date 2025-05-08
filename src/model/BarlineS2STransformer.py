# sequence to sequence transformer that takes adds barlines to a barline-less sequence
import torch
import torch.nn as nn
from .PositionalEncoding import PositionalEncoding

class BarlineS2STransformer(nn.Module):
    """
    Transformer-based model to add barlines to a barline-less sequence. 
    Input is a sequence of tokens, and output is a sequence of tokens with barlines added.
    """
    
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, max_len=5000, bias_init=None):
        super(BarlineS2STransformer, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_len = max_len
        self.device = None
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.positional_encoding = PositionalEncoding(d_model=embed_size, max_len=max_len)
        self.transformer = nn.Transformer(d_model=embed_size, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(embed_size, vocab_size)
        if bias_init is not None:
            self.fc.bias.data = torch.tensor(bias_init, dtype=torch.float32)
        
    def forward(self, src, tgt, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        src: (batch_size, src_seq_length)
        tgt: (batch_size, tgt_seq_length)
        """
        # src and tgt are expected to be of shape (batch_size, seq_length)
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = self.positional_encoding(src, batch_first=True)
        tgt = self.positional_encoding(tgt, batch_first=True)
        # print(src_key_padding_mask)
        # pass through transformer
        tgt_is_causal = tgt_mask is not None
        # print("src", src)
        out1 = self.transformer(src, tgt, tgt_mask=tgt_mask, tgt_is_causal=tgt_is_causal, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask) #, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask) #, tgt_mask=tgt_mask, tgt_is_causal=tgt_is_causal)
        # print("tf", out)
        out = self.fc(out1)
        # print("fc", out)
        return out
    
    def to(self, device):
        super(BarlineS2STransformer, self).to(device)
        self.device = device
        self.embedding = self.embedding.to(device)
        self.positional_encoding = self.positional_encoding.to(device)
        self.transformer = self.transformer.to(device)
        self.fc = self.fc.to(device)
        return self