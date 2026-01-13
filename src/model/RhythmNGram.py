import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm


class RhythmNGram(nn.Module):
    """
    Up-to-n n-gram LM with pure backoff:
      use longest available context; if missing, drop oldest token and retry.

    Stores per-context log-prob vectors as torch.FloatTensor(V).
    """

    def __init__(self, vocab_size: int, order: int, device: str = "cpu"):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.order = int(order)
        self.device = torch.device(device)

        # logprob_tables[k]: dict mapping context-key -> torch.FloatTensor(V)
        # where k is context length (0..order-1). k=0 is unigram with a single key.
        self.logprob_tables: List[Dict[int, torch.Tensor]] = [dict() for _ in range(self.order)]
        self._powers = [self.vocab_size ** i for i in range(self.order)]  # up to k

    def to(self, device):
        self.device = torch.device(device)
        # move stored tensors
        for k in range(self.order):
            for key, vec in self.logprob_tables[k].items():
                self.logprob_tables[k][key] = vec.to(self.device)
        return self

    # ---------- key encoding ----------
    def _ctx_key(self, ctx: Sequence[int]) -> int:
        """
        Encodes a context (length k) into an integer key in base V.
        ctx is ordered oldest->newest.
        """
        V = self.vocab_size
        key = 0
        for t in ctx:
            key = key * V + int(t)
        return key

    # ---------- training ----------
    @classmethod
    def train_from_tokenized_dataset(
        cls,
        tokenized_dataset_path: Union[str, Path],
        *,
        vocab_size: int = 60,
        order: int = 5,
        alpha: float = 0.1,
        split_key: str = "split",
        train_split_value: str = "train",
        sequence_keys: Optional[Sequence[str]] = None,  # e.g. ["0","1","2","3"]
        device: str = "cpu",
    ) -> "RhythmNGram":
        """
        Reads your JSON dict-of-dicts and trains backoff n-gram tables.
        Smoothing: add-alpha (Lidstone) per context.
        Backoff: strict (no interpolation).
        """
        tokenized_dataset_path = Path(tokenized_dataset_path)
        with tokenized_dataset_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        model = cls(vocab_size=vocab_size, order=order, device=device)

        # counts[k][key] -> np.ndarray(V,) counts of next token given k-length context
        counts: List[Dict[int, np.ndarray]] = [dict() for _ in range(order)]
        V = vocab_size

        def bump(k: int, key: int, nxt: int):
            d = counts[k]
            arr = d.get(key)
            if arr is None:
                arr = np.zeros(V, dtype=np.int64)
                d[key] = arr
            arr[nxt] += 1

        for _, item in tqdm(data.items(), desc="Training RhythmNGram"):
            if item.get(split_key) != train_split_value:
                continue

            for sk, seq in item.items():
                if sk == split_key:
                    continue
                if sequence_keys is not None and sk not in sequence_keys:
                    continue
                if not isinstance(seq, list) or len(seq) < 2:
                    continue

                # tokens are strings like "2"; cast to int
                toks = [int(t) for t in seq]

                # for each position i, we predict token toks[i] from preceding context
                # i.e. model learns P(toks[i] | toks[i-k:i]) for k=0..order-1
                for i in range(len(toks)):
                    nxt = toks[i]
                    # contexts of length k use preceding k tokens
                    for k in range(order):
                        if k == 0:
                            key = 0  # single unigram bucket
                            bump(0, key, nxt)
                        else:
                            if i - k < 0:
                                continue
                            ctx = toks[i - k : i]  # length k
                            key = model._ctx_key(ctx)
                            bump(k, key, nxt)

        # Convert counts -> logprob tables with add-alpha smoothing
        for k in range(order):
            model.logprob_tables[k] = {}
            for key, arr in counts[k].items():
                denom = float(arr.sum()) + alpha * V
                probs = (arr.astype(np.float32) + alpha) / denom
                logprobs = np.log(probs, dtype=np.float32)
                model.logprob_tables[k][key] = torch.from_numpy(logprobs).to(model.device)

        # Ensure unigram fallback exists even if something went wrong
        if 0 not in model.logprob_tables[0]:
            # uniform fallback
            model.logprob_tables[0][0] = torch.full((V,), -np.log(V), device=model.device)

        train_seqs: List[List[int]] = []
        val_seqs: List[List[int]] = []

        for _, item in data.items():
            split = item.get(split_key)
            for sk, seq in item.items():
                if sk == split_key:
                    continue
                if sequence_keys is not None and sk not in sequence_keys:
                    continue
                if not isinstance(seq, list) or len(seq) < 1:
                    continue

                toks = [int(t) for t in seq]
                if split == train_split_value:
                    train_seqs.append(toks)
                elif split == "val":
                    val_seqs.append(toks)

        # --- after you finish building model.logprob_tables[...] (i.e., after counts->logprobs) ---
        def _perplexity_on_sequences(seqs: List[List[int]]) -> float:
            if not seqs:
                return float("nan")

            V = model.vocab_size
            max_k = model.order - 1

            total_nll = 0.0
            total_tokens = 0

            # compute NLL over all tokens: -log P(x_i | x_<i) with backoff
            for toks in seqs:
                L = len(toks)
                if L == 0:
                    continue

                for i in range(L):
                    nxt = toks[i]
                    found = False

                    # try longest available context up to n-1, then back off
                    for k in range(min(max_k, i), 0, -1):
                        ctx = toks[i - k : i]  # preceding k tokens
                        key = model._ctx_key(ctx)
                        vec = model.logprob_tables[k].get(key)
                        if vec is not None:
                            total_nll -= float(vec[nxt].item())
                            found = True
                            break

                    if not found:
                        total_nll -= float(model.logprob_tables[0][0][nxt].item())

                    total_tokens += 1

            if total_tokens == 0:
                return float("nan")

            return float(np.exp(total_nll / total_tokens))

        train_ppl = _perplexity_on_sequences(train_seqs)
        val_ppl = _perplexity_on_sequences(val_seqs)

        # change the return line from:
        #   return model
        # to:
        return model, train_ppl, val_ppl

    # ---------- inference ----------
    @torch.no_grad()
    def forward_last(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns log-probs for next token given the last (non-pad) context in each batch item.

        x: LongTensor (B,T) or (T,)
        lengths: LongTensor (B,) true lengths if padded. If None, assumes full length T.
        output: FloatTensor (B,V) log-probs
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        B, T = x.shape
        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=x.device)

        x_cpu = x.detach().cpu().numpy()
        lengths_cpu = lengths.detach().cpu().numpy()

        V = self.vocab_size
        max_k = self.order - 1
        out = torch.empty((B, V), dtype=torch.float32, device=self.device)

        for b in range(B):
            L = int(lengths_cpu[b])
            prefix = x_cpu[b, :L].tolist() if L > 0 else []

            found = False
            # try k = max_k down to 1, then unigram k=0
            for k in range(min(max_k, len(prefix)), 0, -1):
                ctx = prefix[-k:]  # last k tokens
                key = self._ctx_key(ctx)
                vec = self.logprob_tables[k].get(key)
                if vec is not None:
                    out[b] = vec
                    found = True
                    break

            if not found:
                out[b] = self.logprob_tables[0][0]  # unigram fallback

        return out

    def forward(self, x, hidden=None, c=None, batched: bool = False, lengths: Optional[torch.Tensor] = None):
        """
        Compatibility: returns (B,1,V) and None for hidden.
        """
        lp = self.forward_last(x, lengths=lengths)  # (B,V)
        return lp.unsqueeze(1), None  # (B,1,V), hidden placeholder
    
    # ---------- save/load ----------
    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to disk so it can be loaded in another file/process.
        """
        path = Path(path)

        payload = {
            "vocab_size": self.vocab_size,
            "order": self.order,
            "logprob_tables": [
                {key: vec.detach().cpu() for key, vec in table.items()}
                for table in self.logprob_tables
            ],
        }

        torch.save(payload, path)

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cpu") -> "BackoffNGramLM":
        """
        Load a saved BackoffNGramLM.
        """
        path = Path(path)
        payload = torch.load(path, map_location="cpu")

        model = cls(
            vocab_size=payload["vocab_size"],
            order=payload["order"],
            device=device,
        )

        model.logprob_tables = [
            {int(key): vec.to(model.device) for key, vec in table.items()}
            for table in payload["logprob_tables"]
        ]

        return model