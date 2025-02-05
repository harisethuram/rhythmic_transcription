# determines portion of a note that is sound vs rest using a beta distribution
from typing import List, Tuple
import torch
from torch.distributions import Beta
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

from .model_utils import get_beta_params_from_mode_and_spread, beta_pdf

class BetaChannel(nn.Module):
    def __init__(self, modes: List, spreads: List):
        """
        params: list of tuples of (alpha, beta) for the beta distribution
        """
        super(BetaChannel, self).__init__()
        self.modes = modes
        self.spreads = spreads
        self.params = [get_beta_params_from_mode_and_spread(float(mode), float(spread)) for mode, spread in zip(modes, spreads)]
        # self.beta_dists = [Beta(alpha, beta) for alpha, beta in self.params]
        
    def forward(self, x):
        """
        x: (...) tensor of floats between 0 and 1
        returns: (..., len(params)) tensor of floats between 0 and 1 consisting of the probabilities of each beta distribution
        """
        
        return torch.stack([beta_pdf(x, alpha, beta) for alpha, beta in self.params], dim=-1)

    def plot(self, path: str):
        """
        Plot the beta distributions and save the plot to path.
        """
        import matplotlib.pyplot as plt
        x = torch.linspace(0, 1, 1000)
        for alpha, beta in self.params:
            plt.plot(x, beta_pdf(x, alpha, beta))
        plt.legend([f"mode: {mode}, spread: {spread}" for mode, spread in zip(self.modes, self.spreads)])
        plt.savefig(path)
        plt.clf()
        
