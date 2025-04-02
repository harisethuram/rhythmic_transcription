# fit the beta channel model given modes and spreads and save it as .pth file

import argparse
import torch
from fractions import Fraction
import os

from src.model.BetaChannel import BetaChannel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", type=str, required=True, help="Comma-separated list of modes for the Beta distributions")
    parser.add_argument("--spreads", type=str, required=True, help="Comma-separated list of spreads for the Beta distributions")
    parser.add_argument("--output_dir", type=str, required=True, help="Dir to save the model")
    args = parser.parse_args()
    
    modes = [Fraction(mode) for mode in args.modes.split(",")]
    spreads = [Fraction(spread) for spread in args.spreads.split(",")]
    
    beta_channel = BetaChannel(modes, spreads)
    os.makedirs(args.output_dir, exist_ok=True)
    
    torch.save(beta_channel, os.path.join(args.output_dir, "beta_channel.pth"))
    beta_channel.plot(os.path.join(args.output_dir, "beta_channel.png"))