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
from src.const_tokens import *

def step(model, input_data, target_data, src_key_padding_mask, tgt_key_padding_mask, criterion, optimizer, val=False):
    # causal_mask = nn.Transformer.generate_square_subsequent_mask(target_data.size(1)).to(model.device).unsqueeze(0).expand(input_data.size(0) * model.num_heads, -1, -1)
    
    target_data_1 = target_data[:, :-1]
    target_data_2 = target_data[:, 1:]
    tgt_key_padding_mask_1 = tgt_key_padding_mask[:, :-1]
    tgt_key_padding_mask_2 = tgt_key_padding_mask[:, 1:]
    
    causal_mask = model.transformer.generate_square_subsequent_mask(target_data_1.size(1)).to(model.device)#.unsqueeze(0).expand(input_data.size(0) * model.num_heads, -1, -1)
    
    # print("causal_mask:")
    # print(causal_mask)
    # input()
    # Forward pass
    if not val:
        optimizer.zero_grad()
    outputs = model(
        src=input_data,
        tgt=target_data_1,
        tgt_mask=causal_mask,
        src_key_padding_mask=src_key_padding_mask,
        tgt_key_padding_mask=tgt_key_padding_mask_1
    )
    # print(target_data_2.shape)
    loss = criterion(outputs.reshape(-1, outputs.size(-1)), target_data_2.reshape(-1))
    # print("loss: ", loss.item(), outputs.reshape(-1, outputs.size(-1)).shape, target_data_2.reshape(-1).shape)
    # print(outputs.reshape(-1, outputs.size(-1))[:10], target_data_2.reshape(-1)[:10])
    # input()
    if not val:
        loss.backward()
        optimizer.step()
    return loss.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Barline S2S Transformer model")
    parser.add_argument("--data_dir", type=str, help="Directory containing the tokenized parallel data")
    parser.add_argument("--output_dir", type=str, help="Directory to save the trained model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension of the transformer model")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads in the transformer model")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers in the transformer model")
    parser.add_argument("--max_len", type=int, default=5000, help="Maximum length of the input sequences")
    
    args = parser.parse_args()
    print("HYPERPARAMETERS:")
    print(f"lr: {args.learning_rate}, batch_size: {args.batch_size}, hidden_dim: {args.hidden_dim}, num_heads: {args.num_heads}, num_layers: {args.num_layers}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load the data
    print("Loading data...")
    train_loader, val_loader, token_to_id, id_to_token = preprocess_parallel_data(args.data_dir, args.batch_size, DEVICE)

    # Initialize the model
    print("Initializing model...")
    model = BarlineS2STransformer(
        vocab_size=len(token_to_id),
        embed_size=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_len=args.max_len
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
            running_val_loss += step(model, input_data, target_data, src_key_padding_mask, tgt_key_padding_mask, criterion, optimizer, val=True)
            
        running_val_loss /= len(val_loader)
        loss_vals[0] = {"train": running_train_loss, "val": running_val_loss}
    print("Initial loss: ", loss_vals[0])
    best_model = model.state_dict()
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
        with torch.no_grad():
            for _, input_data, target_data, src_key_padding_mask, tgt_key_padding_mask in val_loader:
                running_val_loss += step(model, input_data, target_data, src_key_padding_mask, tgt_key_padding_mask, criterion, optimizer, val=True)
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