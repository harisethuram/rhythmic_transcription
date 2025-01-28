import argparse
import torch
from fractions import Fraction

from src.model.BetaChannel import BetaChannel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", type=str, required=True, help="Comma-separated list of modes for the Beta distributions")
    parser.add_argument("--spreads", type=str, required=True, help="Comma-separated list of spreads for the Beta distributions")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the model")
    args = parser.parse_args()
    
    modes = [float(Fraction(mode)) for mode in args.modes.split(",")]
    spreads = [float(Fraction(spread)) for spread in args.spreads.split(",")]
    
    beta_channel = BetaChannel(modes, spreads)
        
    beta_channel.save(args.output_path)