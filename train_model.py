import argparse
import os
import torch
import torch.nn as nn
import pickle as pkl
from tqdm import tqdm

from src.model.model import RhythmLSTM
from src.preprocess.collate import preprocess_data

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
        
    train_loader, val_loader, token_to_id, id_to_token = preprocess_data(args.processed_data_dir, args.batch_size, device=device)
    
    # Initialize the model
    vocab_size = len(token_to_id.keys())
    model = RhythmLSTM(vocab_size, args.embed_size, args.hidden_size, args.num_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    
    model.train()
    