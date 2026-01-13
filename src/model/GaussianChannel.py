# model computes probability of note given symbollic duration, not used at the moment
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
    
class GaussianChannel(nn.Module):
    def __init__(self, sigma=0.1):
        super(GaussianChannel, self).__init__()
        self.sigma = sigma
        self.mu = 1
    

    def forward(self, input_duration: torch.Tensor, symbollic_duration: float, tempo: float):
        if symbollic_duration != 0:
            mean = self.mu * symbollic_duration
            std = self.sigma * symbollic_duration            
        else: 
            mean = 0
            std = 0.1

        input_symbollic_duration = input_duration * tempo / 60
        probs = torch.exp(-0.5 * (input_symbollic_duration - mean) ** 2 / (std ** 2))
        return probs
    
    def plot(self):
        # Plot Gaussian distribution for different symbollic durations
        x = torch.linspace(-1, 6, 1000)
        for symbollic_duration in [0.5, 1, 2, 4]:
            if symbollic_duration != 0:
                mean = self.mu * symbollic_duration
                std = self.sigma * symbollic_duration            
            else: 
                mean = 0
                std = 0.1
            y = torch.exp(-0.5 * (x - mean) ** 2 / (std ** 2))
            plt.plot(x.numpy(), y.numpy(), label=f'Symbollic Duration: {symbollic_duration}')
        plt.title('Gaussian Channel Probability Distribution, tempo=60 BPM')
        plt.xlabel('Input Duration')
        plt.ylabel('Probability')
        plt.legend()
        plt.savefig('gaussian_channel_probability_distribution.png')
        
if __name__ == "__main__":
    # # input_duration = torch.Tensor([1, 2, 3])
    # symbollic_duration = torch.Tensor([1, 2, 3])
    model = GaussianChannel()
    model.plot()
    # probs = model(input_duration, symbollic_duration)
    # print(probs)