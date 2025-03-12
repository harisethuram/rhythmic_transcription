import numpy as np

def get_regularize_fn(alpha, gamma=1, l=1):
    def reg_fn(alpha):
        return gamma * sum(abs(alpha) ** l)
    return reg_fn

def get_sin_loss_fn(eta=1, gamma=0, asymp=0, reg_fn):
    """
    Returns a parameterized sin loss function
    eta: scale factor for asymptote part of the loss
    gamma: scale factor for the regularization term
    asymp: must be in [0, 1) to specify the asymptote value
    
    """
    if asymp < 0 or asymp >= 1:
        raise ValueError("Asymptote must be in [0, 1)")

    def sin_loss(alpha, x):
        """
        x: np array of shape (n)
        """
        def inv(y):
            return (y - 1) ** 2 / (y - asymp)
        
        x = x.copy() * alpha[0]
            
        lesser_than_one = x < 1
        x[lesser_than_one] = inv(x[lesser_than_one]) * eta
        
        x[np.invert(lesser_than_one)] = np.sin(np.pi * x[np.invert(lesser_than_one)]) ** 2 
            
        val = x.mean() + reg_fn(alpha)
        return val
    
    return sin_loss