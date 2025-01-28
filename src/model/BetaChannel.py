# determines portion of a note that is sound vs rest using a beta distribution
from typing import List, Tuple
import torch
from torch.distributions import Beta
import torch.nn as nn

from .model_utils import get_beta_params_from_mode_and_spread

class BetaChannel(nn.Module):
    def __init__(self, modes: List[float], spreads: List[float]):
        """
        params: list of tuples of (alpha, beta) for the beta distribution
        """
        super(BetaChannel, self).__init__()
        self.modes = modes
        self.spreads = spreads
        self.params = [get_beta_params_from_mode_and_spread(mode, spread) for mode, spread in zip(modes, spreads)]
        self.beta_dist = Beta
        
    def forward(self, x):
        """
        x: (...) tensor of floats between 0 and 1
        returns: (..., len(params)) tensor of floats between 0 and 1 consisting of the probabilities of each beta distribution
        """
        
        return torch.stack([self.beta_dist(alpha, beta).log_prob(x).exp() for alpha, beta in self.params], dim=-1)
