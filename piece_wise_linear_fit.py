# model that takes in a sequence of float onset times, and fits a piece wise linear function to it

import numpy as np
from scipy.optimize import minimize, dual_annealing
from tqdm import tqdm
import argparse
import os
import json


from src.preprocess.ingestion_utils import get_performance_onsets, get_score_onsets, get_score_note_lengths, get_performance_note_lengths, get_performance_pitches
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
    
    if lbda >= 10:
        # if lbda is large, we likely only need one segment, so don't need to forgo the additional computation
        alpha, error = find_optimal_alpha(onset_lengths, loss_fn, gamma=gamma)
        seg_idx = len(onset_lengths)
        return [seg_idx], [alpha]
    
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
    print("Score info:", args.score_path, args.score_part_number)
    print("eval:", args.eval)
    
    print("Starting")
    
    if args.loss_fn == "sin_loss":
        loss_fn = sin_loss
    elif args.loss_fn == "round_loss":
        loss_fn = round_loss
    else:
        raise ValueError("Invalid loss function")
    
    performance_onsets = get_performance_onsets(args.performance_path)
    print("performance len:", len(performance_onsets))
    
    raw_performance_onset_lengths = np.diff(performance_onsets)[:args.test_len]
        
    segments, alphas = piecewise_fit(raw_performance_onset_lengths, loss_fn, args.lbda, gamma=args.gamma, debug=args.debug)
    
    scaled_performance_onset_lengths, split_onset_lengths = scale_onsets(raw_performance_onset_lengths, segments, alphas)
    
    os.makedirs(args.output_dir, exist_ok=True)
    plt_path = os.path.join(args.output_dir, "piecewise_fit.png")
    
    # evaluation
    if args.eval:
        
        raw_score_onsets = get_score_onsets(args.score_path)[args.score_part_number-1]
        raw_score_onset_lengths = np.diff(raw_score_onsets)[:args.test_len]    
        print("score len:", len(raw_score_onsets))  
        
        error, scaled_performance_onset_times, scaled_score_onset_times, scaled_score_onset_lengths, alignment = evaluate_onset_trascription(scaled_performance_onset_lengths, raw_score_onset_lengths)
        
        print("Error:", error)
        # plot only the first 20 onsets
        plot_onset_times(scaled_performance_onset_times, raw_performance_onset_lengths, scaled_score_onset_times, raw_score_onset_lengths, alignment, plt_path)
        
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump({"error": str(error), "prediction": scaled_performance_onset_times.tolist(), "ground_truth": scaled_score_onset_times.tolist()}, f)
    
    # convert onsets to note and rest durations
    
    performance_note_info = get_performance_note_lengths(args.performance_path)
    performance_note_lengths = performance_note_info[:-1,0]
    performance_rest_lengths = performance_note_info[:-1,1]
    total_lengths = performance_note_lengths + performance_rest_lengths
    standardized_performance_note_lengths = performance_note_lengths / total_lengths
    standardized_performance_rest_lengths = performance_rest_lengths / total_lengths
    
    pitches = get_performance_pitches(args.performance_path)[:-1]
    # print("Pitches:", pitches)
    
    
    # scale the note lengths
    # scaled_note_lengths, _ = scale_onsets(performance_note_lengths, segments, alphas)
    # scaled_rest_lengths = scaled_performance_onset_lengths - scaled_note_lengths[:-1]
    # print("Scaled Onset lenghts:", scaled_performance_onset_lengths)
    # print("Standardized Note lengths:", standardized_performance_note_lengths)
    # print("Standardized note lengths:", standardized_performance_rest_lengths)
    all_info = np.vstack((scaled_performance_onset_lengths, standardized_performance_note_lengths, standardized_performance_rest_lengths, pitches)).T.tolist()
    # print(all_info)
    with open(os.path.join(args.output_dir, "note_info.json"), "w") as f:
        json.dump(all_info, f)
    # print("Rest lengths:", performance_rest_lengths)
    # print("Scaled rest lengths:", scaled_rest_lengths)
        
    # all_score_note_length_info = get_score_note_lengths(args.score_path, args.score_part_number)
    # print("Score note lengths:", all_score_note_length_info)

    # input() 
    
    