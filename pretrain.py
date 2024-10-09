import argparse
import os
import torch
import torch.nn as nn
import pickle as pkl
from tqdm import tqdm
import wandb
import json

from src.model.model import RhythmLSTM
from src.preprocess.collate import preprocess_data

def step(model, data, criterion, optimizer=None, val=False):
    data = data.to(device)
    targets = data[:, 1:]
    data = data[:, :-1]

    # Forward pass
    if not val:
        optimizer.zero_grad()
    outputs, _ = model(data)

    loss = criterion(outputs, targets.reshape(-1))
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad_token_id = 0
        
    # Load the processed data
    print("Loading data...")
    train_loader, val_loader, token_to_id, id_to_token = preprocess_data(args.processed_data_dir, args.batch_size, device=device)
    
    # Initialize the model
    print("Initializing model...")
    vocab_size = len(token_to_id.keys())
    model = RhythmLSTM(vocab_size, args.embed_size, args.hidden_size, args.num_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print("Training model...")
    loss_vals = {}
    
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
        
    for epoch in range(args.num_epochs):
        pbar = tqdm(train_loader)   
            
        model.train()
        running_train_loss = 0.0
        for batch_idx, (file_names, data) in enumerate(pbar):
            loss = step(model, data, criterion, optimizer=optimizer)
            running_train_loss += loss
            pbar.set_description(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {loss}")
        running_train_loss /= len(train_loader)
        
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():            
            for i, (file_names, data) in enumerate(val_loader):
                running_val_loss += step(model, next(iter(val_loader))[1], criterion, val=True)
            running_val_loss /= len(val_loader)
        
        loss_vals[epoch+1] = {"train": running_train_loss, "val": running_val_loss}
        print("Epoch {}/{}: Avg Train Loss: {}, Avg Val Loss: {}".format(epoch+1, args.num_epochs, running_train_loss, running_val_loss))
        
    print("Saving...")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pth"))    
    
    with open(os.path.join(args.output_dir, "loss_vals.json"), "w") as f:
        json.dump(loss_vals, f, indent=4)
            
            
            