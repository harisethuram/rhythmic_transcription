import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import os
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from src.preprocess.collate import preprocess_parallel_data
from src.model.BarlineS2STransformer import BarlineS2STransformer
from src.model.BarlineS2SLSTM import BarlineS2SLSTM
from src.const_tokens import *
from src.utils import open_processed_data_dir

def step(model, input_data, target_data, src_key_padding_mask, tgt_key_padding_mask, criterion, optimizer, val=False, print_logits=False):    
    target_data_1 = target_data[:, :-1]
    target_data_2 = target_data[:, 1:]
    tgt_key_padding_mask_1 = tgt_key_padding_mask[:, :-1]
    tgt_key_padding_mask_2 = tgt_key_padding_mask[:, 1:]
    
    
    # causal_mask = None
    # print(target_data_1, target_data_2, tgt_key_padding_mask_1)
    # input()
    # print("causal_mask:")
    # print(causal_mask)
    # input()
    # Forward pass
    if not val:
        optimizer.zero_grad()
        
    if isinstance(model, BarlineS2STransformer):
        causal_mask = model.transformer.generate_square_subsequent_mask(target_data_1.size(1)).to(model.device)#.expand(input_data.size(0) * model.num_heads, -1, -1)
        outputs = model(
            src=input_data,
            tgt=target_data_1,
            tgt_mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask_1
        )
    else:
        outputs, _, _ = model(
            src=input_data,
            tgt=target_data_1,
        )
    
    if print_logits:
        print("outputs: ", outputs[..., :10, :10])
        input()
    
    # print(target_data_2.shape)
    loss = criterion(outputs.reshape(-1, outputs.size(-1)), target_data_2.reshape(-1))
    if not val:
        loss.backward()
        optimizer.step()
    return loss.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Barline S2S LSTM or Transformer model")
    parser.add_argument("--want_transformer", action="store_true", help="Use transformer instead of LSTM, default is LSTM")
    parser.add_argument("--data_dir", type=str, help="Directory containing the tokenized parallel data")
    parser.add_argument("--output_dir", type=str, help="Directory to save the trained model")
    parser.add_argument("--barline_data_dir", type=str, default=None, help="Directory containing the barline data")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension of the transformer model")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads in the transformer model")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers in the transformer model")
    parser.add_argument("--max_len", type=int, default=5000, help="Maximum length of the input sequences")
    
    args = parser.parse_args()
    print("ARGS:")
    print(args)
    # print("HYPERPARAMETERS:")
    # print(f"lr: {args.learning_rate}, batch_size: {args.batch_size}, hidden_dim: {args.hidden_dim}, num_heads: {args.num_heads}, num_layers: {args.num_layers}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load the data
    print("Loading data...")
    train_loader, val_loader, token_to_id, id_to_token = preprocess_parallel_data(args.data_dir, args.batch_size, DEVICE)
    
    # Load the barline data if provided
    bias_init = None
    if args.barline_data_dir is not None:
        with open(os.path.join(args.barline_data_dir, "train_frequencies.json"), "r") as f:
            freq_dict = json.load(f)
        
        freq_dict = {v: int(freq_dict.get(str(k), 0)) for k, v in token_to_id.items()}
        freq_list = [(k, v) for k, v in freq_dict.items()]
        freq_list.sort(key=lambda x: x[0], reverse=True)
        bias_init = torch.Tensor([x[1] for x in freq_list])
        bias_init = bias_init / bias_init.sum()
            
    print("Bias init: ", bias_init) 
        
        

    # Initialize the model
    print("Initializing model...")
    if args.want_transformer:
        model = BarlineS2STransformer(
            vocab_size=len(token_to_id),
            embed_size=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            max_len=args.max_len,
            bias_init=bias_init
        ).to(DEVICE)
    else:
        model = BarlineS2SLSTM(
            vocab_size=len(token_to_id),
            embed_size=args.hidden_dim,
            num_layers=args.num_layers,
            bias_init=bias_init
        ).to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=token_to_id[PADDING_TOKEN])
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print("Training model...")
    loss_vals = {"best_val_epoch": None, "best_val_loss": float("inf")}
    # model.eval()
    
    # Initial loss:
    print("Calculating initial loss...")
    with torch.no_grad():
        running_train_loss = 0.0
        for _, input_data, target_data, src_key_padding_mask, tgt_key_padding_mask in tqdm(train_loader):
            running_train_loss += step(model, input_data, target_data, src_key_padding_mask, tgt_key_padding_mask, criterion, optimizer, val=True) 
            
        running_train_loss /= len(train_loader)
        
        running_val_loss = 0.0
        for _, input_data, target_data, src_key_padding_mask, tgt_key_padding_mask in tqdm(val_loader):
            running_val_loss += step(model, input_data, target_data, src_key_padding_mask, tgt_key_padding_mask, criterion, optimizer, val=True, print_logits=False)
            
        running_val_loss /= len(val_loader)
        loss_vals[0] = {"train": running_train_loss, "val": running_val_loss}
    print("Initial loss: ", loss_vals[0])
    
    best_val_loss = float("inf")
    pbar = tqdm(range(args.num_epochs))
    print("Starting training...")
    for epoch in pbar:
        model.train()
        running_train_loss = 0.0
        
        for _, input_data, target_data, src_key_padding_mask, tgt_key_padding_mask in tqdm(train_loader):
            loss = step(model, input_data, target_data, src_key_padding_mask, tgt_key_padding_mask, criterion, optimizer)
            running_train_loss += loss
        running_train_loss /= len(train_loader)
        
        model.eval()
        running_val_loss = 0.0
        print_logit = epoch % 5 == 0
        with torch.no_grad():
            tmp_count = 0
            for _, input_data, target_data, src_key_padding_mask, tgt_key_padding_mask in val_loader:
                print_and = tmp_count == 0
                tmp_count += 1
                running_val_loss += step(model, input_data, target_data, src_key_padding_mask, tgt_key_padding_mask, criterion, optimizer, val=True, print_logits=False)#(print_logit and print_and))
            running_val_loss /= len(val_loader)
        loss_vals[epoch+1] = {"train": running_train_loss, "val": running_val_loss}
        print("Epoch {}: Train Loss: {}, Val Loss: {}".format(epoch + 1, round(running_train_loss, 3), round(running_val_loss, 3)))
        if running_val_loss < loss_vals["best_val_loss"]:
            loss_vals["best_val_loss"] = running_val_loss
            loss_vals["best_val_epoch"] = epoch + 1
            torch.save(model, os.path.join(args.output_dir, "model.pth"))
            
        pbar.set_description("Avg Train Loss: {}, Avg Val Loss: {}".format(round(running_train_loss, 3), round(running_val_loss, 3)))
        
    print("Training complete. Best validation loss: ", loss_vals["best_val_loss"])
    print("Best validation epoch: ", loss_vals["best_val_epoch"])
    # print("Saving model...")
    # confirm that this is the best model
    model = torch.load(os.path.join(args.output_dir, "model.pth"))

    running_val_loss = 0.0
    with torch.no_grad():
        for _, input_data, target_data, src_key_padding_mask, tgt_key_padding_mask in val_loader:
            running_val_loss += step(model, input_data, target_data, src_key_padding_mask, tgt_key_padding_mask, criterion, optimizer, val=True)
        running_val_loss /= len(val_loader)
    print("best model val loss (loaded and recomputed):", running_val_loss)
    print("Best validation loss (should be the same):", loss_vals["best_val_loss"])
    print("Best validation epoch:", loss_vals["best_val_epoch"])
    
    print("Plotting loss...")
    # plot loss vals
    train_losses = [loss_vals[i]["train"] for i in range(args.num_epochs + 1)]
    val_losses = [loss_vals[i]["val"] for i in range(args.num_epochs + 1)]
    epochs = list(range(args.num_epochs + 1))
    
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss with best val loss: {loss_vals['best_val_loss']:.4f} @ epoch {loss_vals['best_val_epoch']}\nlr: {args.learning_rate}, bz: {args.batch_size}, dim: {args.hidden_dim}, heads: {args.num_heads}, layers: {args.num_layers}")
    plt.savefig(os.path.join(args.output_dir, "loss.png"))
    
    with open(os.path.join(args.output_dir, "loss.json"), "w") as f:
        json.dump(loss_vals, f, indent=4)
    print("Complete.")