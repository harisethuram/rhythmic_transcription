import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.model.RhythmNGram import RhythmNGram
from src.utils import open_processed_data_dir

if __name__ == "__main__":
    data_dir = "processed_data/all/no_barlines"
    token_to_id_new, id_to_token_new, _ = open_processed_data_dir(data_dir)
    print("Training")
    orders = range(1, 15, 2)
    train_ppls = []
    val_ppls = []
    output_dir = Path("models/rhythm_ngram")
    output_dir.mkdir(parents=True, exist_ok=True)
    for order in orders:
        print(f"Order: {order}")
        model, train_ppl, val_ppl = RhythmNGram.train_from_tokenized_dataset(
            tokenized_dataset_path=Path(data_dir) / "tokenized_dataset.json",
            vocab_size=len(token_to_id_new),
            order=order
        )
        print("train ppl:", train_ppl)
        print("val ppl:", val_ppl)
        print("saving")
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)
        model.save("models/rhythm_ngram/rhythm_ngram_model_order_" + str(order) + ".pt")
    # print("Trained RhythmNGram model:", model)
    plt.plot(orders, train_ppls, label="Train PPL")
    plt.plot(orders, val_ppls, label="Validation PPL")
    plt.xlabel("N-gram Order")
    plt.ylabel("Perplexity")
    plt.title("Rhythm N-gram Model Perplexity vs Order")
    plt.legend()
    plt.savefig("models/rhythm_ngram/rhythm_ngram_ppl_vs_order.png")    