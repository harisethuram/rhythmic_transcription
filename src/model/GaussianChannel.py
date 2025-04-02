# model computes probability of note given symbollic duration, not used at the moment
import torch
import torch.nn as nn

class GaussianChannel(nn.Module): 
    def __init__(self, sigma=0.1):
        super(GaussianChannel, self).__init__()
        self.sigma = sigma
        
    def forward(self, input_duration, symbollic_duration, want_log_probs=False):
        """
        Computes the probability of a note given a symbollic duration
        input_duration: torch.Tensor, shape (batch_size)
        symbollic_duration: torch.Tensor, shape (# of symbollic durations)
        return: torch.Tensor, shape (batch_size, # of symbollic durations)
        """ 
        mean_D = 1
        std_D = self.sigma ** 0.5
        
        mean_Dx = torch.Tensor(mean_D * input_duration).unsqueeze(1)
        std_Dx = torch.Tensor(std_D * abs(input_duration)).unsqueeze(1)
        
        Y = symbollic_duration.unsqueeze(0)
        
        dist_Dx = torch.distributions.Normal(mean_Dx, std_Dx)
        
        probs = dist_Dx.log_prob(Y)   
        
        if not want_log_probs:
            probs = probs.exp()
            
        return probs
    
# testing
if __name__ == "__main__":
    input_duration = torch.Tensor([1, 2, 3])
    symbollic_duration = torch.Tensor([1, 2, 3])
    model = GaussianChannel()
    probs = model(input_duration, symbollic_duration)
    print(probs)