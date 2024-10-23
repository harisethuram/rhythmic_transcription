import torch
import pickle as pkl
import torch.nn as nn
import argparse
import music21
import os

from src.model.GaussianChannel import GaussianChannel
from src.model.RhythmLSTM import RhythmLSTM

def text_to_onset_list(path, part=0):
    # convert text of cols (onset_time, pitch, duration) to list of tuples of form (onset_time, duration)
    with open(path, "r") as f:
        lines = f.readlines()
        onset_list = [(float(line.strip().split("\t")[0]), float(line.strip().split("\t")[-1])) for line in lines]
    
    return onset_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--sigma", type=float, default=0.1, help="Standard deviation of the Gaussian noise")
    parser.add_argument("--output_path", type=str, help="Output path")
    parser.add_argument("--input_path", type=str, help="Path to midi to transcribe")
    parser.add_argument("--tokenizer_dir", type=str, help="Path to the tokenizer directory")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = pkl.load(open(args.model_path, "rb"))
    model = torch.load(args.model_path).to(device)
    print(model)

    channel_model = GaussianChannel(args.sigma).to(device)
    
    with torch.no_grad():
        # input is a sequence of tuples of form (onset_time, duration)
        onset_list = torch.Tensor(text_to_onset_list(args.input_path)).to(device)
        id_to_token = pkl.load(open(os.path.join(args.tokenizer_dir, "id_to_token.pkl"), "rb"))
        # for 
        
        