# model that takes in a sequence of float onset times, and fits a piece wise linear function to it

import numpy as np
from scipy.optimize import minimize, dual_annealing
from tqdm import tqdm
import argparse
import os

from src.preprocess.ingestion_utils import get_performance_onsets, get_score_onsets
from src.eval.utils import sin_loss, round_loss, evaluate_onset_trascription, plot_onset_times, scale_onsets

def find_optimal_alpha(onset_lengths, loss_fn, gamma=0, initial_alpha=3.0, upper_bound=20):
    """ Finds the optimal alpha that minimizes the deviation from integer values. 
    onset_lengths: np array of shape (n)
    loss_fn: function that takes in alpha and onset_lengths and returns a scalar
    initial_alpha: initial guess for alpha
    upper_bound: upper bound for alpha
    gamma: alpha regularization term
    """

    initial_alpha = np.array([3.0])
    bounds = [(1e-6, 99)]
    # gamma = 0.01

    result = dual_annealing(
        loss_fn,
        bounds=bounds,
        args=(onset_lengths,gamma)
    )
    final_loss = loss_fn(result.x, onset_lengths, gamma=0)
    # print(onset_lengths, final_loss, result.x)

    return result.x[0], final_loss


def piecewise_fit(onset_lengths, loss_fn, lbda, gamma, debug=False):
    """
    onset_lengths: np array of shape (n)
    loss_fn: function that takes in alpha and onset_lengths and returns a scalar
    lbda: cost of adding a new segment
    """
    n = len(onset_lengths)
    cost = np.inf * np.ones(n+1)
    segmentation = np.zeros(n)
    cost[0] = 0
    alphas = np.zeros(n+1)
    segmentation = np.zeros(n+1)
    
    def reconstruct_segment(segmentation, alphas):
        i = n
        seg_idx = []
        seg_alphas = []
        seg_idx.append(n)
        seg_alphas.append(alphas[i])
        # print(i)
        while segmentation[i] > 0:
            j = segmentation[i]
            print(j)
            seg_idx.append(int(j))
            seg_alphas.append(alphas[int(j)])
            i = int(j)        
        seg_idx.reverse()
        seg_alphas.reverse()
        return seg_idx, seg_alphas
    
    for i in tqdm(range(1, n+1)):
        for j in range(i):
            segment = onset_lengths[j:i]
            alpha, error = find_optimal_alpha(segment, loss_fn, gamma=gamma)
            total_cost = cost[j] + error + lbda
            if total_cost < cost[i]:
                cost[i] = total_cost[0] 
                segmentation[i] = j
                alphas[i] = alpha
            
            if debug:
                print(j, i)
                print("Error:", error)
                print("Total Cost:", total_cost)
                print("Alpha:", alpha)
                print("Segment:", segment)
                print("Scaled Segment:", segment * alpha)
                print("Cost:", cost)
                print("Segmentation:", segmentation)
                print()

    seg_idx, seg_alphas = reconstruct_segment(segmentation, alphas)
    return seg_idx, seg_alphas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--performance_path", type=str, help="File name of the performance onset times")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--score_path", type=str, default=None, help="File name of the score onset times, only required if eval is true")
    parser.add_argument("--score_part_number", type=int, default=None, help="Part number of the score, only required if eval is true")
    parser.add_argument("--loss_fn", type=str, default="sin_loss", help="Loss function to use (sin_loss or round_loss)")
    parser.add_argument("--lbda", type=float, default=0.04, help="Cost of adding a new segment")
    parser.add_argument("--gamma", type=float, default=0.01, help="Regularization term for alpha")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--test_len", type=int, default=100000, help="Number of onsets to limit to for testing")
    
    args = parser.parse_args()
    print("Starting")
    
    if args.loss_fn == "sin_loss":
        loss_fn = sin_loss
    elif args.loss_fn == "round_loss":
        loss_fn = round_loss
    else:
        raise ValueError("Invalid loss function")
    
    performance_onsets = get_performance_onsets(args.performance_path)
    print("performance len:", len(performance_onsets))
    
    performance_onset_lengths = np.diff(performance_onsets)[:args.test_len]
    if args.eval:
        score_onsets = get_score_onsets(args.score_path)[args.score_part_number]
        score_onset_lengths = np.diff(score_onsets)[:args.test_len]    
        print("score len:", len(score_onsets))

    segments, alphas = piecewise_fit(performance_onset_lengths, loss_fn, args.lbda, gamma=args.gamma, debug=args.debug)
    
    scaled_performance_onset_lengths, split_onset_lengths = scale_onsets(performance_onset_lengths, segments, alphas)
        
    print("Splits:")
    for split, alpha in zip(split_onset_lengths, alphas):
        print(split, alpha)
        
    print("Segments:", segments)
    print("Scaled onsets:", scaled_performance_onset_lengths)
    
    rounded_performance_onset_lengths = np.round(scaled_performance_onset_lengths)
    print("Scaled rounded onsets:", rounded_performance_onset_lengths)
    rounded_performance_onset_times = np.array([0] + list(np.cumsum(rounded_performance_onset_lengths)))
    
    os.makedirs(args.output_dir, exist_ok=True)
    plt_path = os.path.join(args.output_dir, "piecewise_fit.png")
    # evaluation
    if args.eval:
        error, scaled_score = evaluate_onset_trascription(rounded_performance_onset_lengths, score_onset_lengths)
        print("Ground truth onsets:", scaled_score)
        print("Error:", error)
        plot_onset_times(rounded_performance_onset_lengths, score_onset_lengths, score_onset_lengths, plt_path)
        
    