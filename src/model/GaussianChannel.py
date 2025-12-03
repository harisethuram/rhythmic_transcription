# model computes probability of note given symbollic duration, not used at the moment
import torch
import torch.nn as nn
import math

    
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
    
if __name__ == "__main__":
    input_duration = torch.Tensor([1, 2, 3])
    symbollic_duration = torch.Tensor([1, 2, 3])
    model = GaussianChannel()
    probs = model(input_duration, symbollic_duration)
    print(probs)