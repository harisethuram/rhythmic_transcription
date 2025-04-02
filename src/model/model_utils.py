import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import beta as beta_dist

def get_beta_params_from_mode_and_spread(mode, spread):
    """
    Find alpha and beta parameters for a Beta distribution given mode and spread.
    
    Parameters:
    - mode: The mode (peak) of the Beta distribution (0 < mode < 1).
    - spread: The standard deviation (spread) of the Beta distribution.
    
    Returns:
    - alpha, beta: The shape parameters of the Beta distribution.
    """
    if not 0 < mode <= 1:
        raise ValueError("Mode must be between 0 and 1.")
    if spread <= 0:
        raise ValueError("Spread must be positive.")
    
    # Define the system of equations
    def equations(p):
        alpha, beta_param = p
        mode_calc = (alpha - 1) / (alpha + beta_param - 2)
        variance = (alpha * beta_param) / ((alpha + beta_param)**2 * (alpha + beta_param + 1))
        spread_calc = np.sqrt(variance)
        return (mode_calc - mode, spread_calc - spread)
    
    # Initial guesses for alpha and beta
    # A common heuristic is to set alpha = beta = 2 for a symmetric distribution
    initial_guess = (2, 2)
    
    # Solve the equations
    solution, infodict, ier, mesg = fsolve(equations, initial_guess, full_output=True)
    if ier != 1:
        raise RuntimeError(f"Could not find a solution: {mesg}")
    
    alpha, beta_param = solution
    
    return alpha, beta_param

def beta_pdf(x, alpha, beta):
    """
    Retuns pdf values of x given a beta distribution parametrized by alpha and beta
    """
    return torch.from_numpy(beta_dist.pdf(x, alpha, beta)).to(x.dtype).to(x.device)