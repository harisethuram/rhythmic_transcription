import argparse
import os
import torch
import torch.nn as nn
import pickle as pkl
from tqdm import tqdm
import json
import matplotlib.pyplot as plt


from src.model.RhythmLSTM import RhythmLSTM
from src.preprocess.collate import preprocess_data
from const_tokens import *

def step(model, data, criterion, optimizer=None, val=False):
    data = data.to(device)
    targets = data[:, 1:]
    data = data[:, :-1]

    # Forward pass
    if not val:
        optimizer.zero_grad()
    outputs, _ = model(data)
    
    loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))
    if not val:
        loss.backward()
        optimizer.step()
    
    return loss.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data_dir", type=str, help="Directory containing the processed data. Has tokenized data, token_to_id, and id_to_token")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--embed_size", type=int, default=256, help="Embedding size")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    
    args = parser.parse_args()
    print(f"lr: {args.learning_rate}, batch_size: {args.batch_size}, embed_size: {args.embed_size}, hidden_size: {args.hidden_size}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(0)
    # Load the processed data
    print("Loading data...")
    train_loader, val_loader, token_to_id, id_to_token = preprocess_data(args.processed_data_dir, args.batch_size, device=device)
    
    # Initialize the model
    print("Initializing model...")
    vocab_size = len(token_to_id.keys())
    model = RhythmLSTM(vocab_size, args.embed_size, args.hidden_size, args.num_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=token_to_id[PADDING_TOKEN])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print("Training model...")
    loss_vals = {"best_val_epoch": None, "best_val_loss": float("inf")}
    
    # initial loss: 
    with torch.no_grad():
        running_train_loss = 0.0
        for _, data in train_loader:
            running_train_loss += step(model, data, criterion, val=True)
            
        running_train_loss /= len(train_loader)
        
        
        running_val_loss = 0.0
        for _, data in val_loader:
            running_val_loss += step(model, data, criterion, val=True)
            
        running_val_loss /= len(val_loader)
        loss_vals[0] = {"train": running_train_loss, "val": running_val_loss}
    
    best_model = model.state_dict()
    best_val_loss = float("inf")
    pbar = tqdm(range(args.num_epochs))
    for epoch in pbar:
            
        model.train()
        running_train_loss = 0.0
        for batch_idx, (file_names, data) in enumerate(train_loader):
            loss = step(model, data, criterion, optimizer=optimizer)
            running_train_loss += loss

        running_train_loss /= len(train_loader)
        
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for i, (file_names, data) in enumerate(val_loader):
                running_val_loss += step(model, next(iter(val_loader))[1], criterion, val=True)
            running_val_loss /= len(val_loader)
        
        loss_vals[epoch+1] = {"train": running_train_loss, "val": running_val_loss}
        
        if running_val_loss < loss_vals["best_val_loss"]:
            loss_vals["best_val_loss"] = running_val_loss
            loss_vals["best_val_epoch"] = epoch+1
            torch.save(model, os.path.join(args.output_dir, "model.pth"))
            
        pbar.set_description("Avg Train Loss: {}, Avg Val Loss: {}".format(round(running_train_loss, 3), round(running_val_loss, 3)))
    

    # confirm that this is the best model
    model = torch.load(os.path.join(args.output_dir, "model.pth"))
    model.eval()

    
    running_val_loss = 0.0
    with torch.no_grad():
        for i, (file_names, data) in enumerate(val_loader):
            running_val_loss += step(model, next(iter(val_loader))[1], criterion, val=True)
        running_val_loss /= len(val_loader)
    print("best model val loss (recomputed):", running_val_loss)
    print("Best validation loss (should be the same):", loss_vals["best_val_loss"])
    print("Best validation epoch:", loss_vals["best_val_epoch"])
    
    
    
    train_losses = [loss_vals[k]["train"] for k in loss_vals if isinstance(k, int)]
    val_losses = [loss_vals[k]["val"] for k in loss_vals if isinstance(k, int)]
    plt.plot(range(len(train_losses)), train_losses, label="Train Loss")
    plt.plot(range(len(val_losses)), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Plot, best val loss: {loss_vals['best_val_loss']:.4f} @ epoch {loss_vals['best_val_epoch']}")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "loss_plot.png"))
    
    with open(os.path.join(args.output_dir, "loss_vals.json"), "w") as f:
        json.dump(loss_vals, f, indent=4)            