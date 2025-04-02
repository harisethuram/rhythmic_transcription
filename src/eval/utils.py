import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import os
import math
from scipy import stats
from tslearn.metrics import dtw_path


def plot_onset_times(prediction, raw_prediction_labels=None, ground_truth=None, raw_score_onset_lengths=None, alignment=None, save_path=None):
    """
    Create the plot of the onset times, potentially with a visualization of the alignment with the ground-truth if those are provided. 
    """
    # Create a new figure
    # 1: 1, ratio between length of predictions and length of image
    l = min(len(prediction), 655)
    # print("figure size:", l, 4)
    # input()
    plt.figure(figsize=(l, 4))
    t = 1
    vertices = [
        (0, 0), (t*2, t), (0, 2*t),  # Define a triangle-like shape
        (0, 0)  # Close the path
    ]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    custom_marker = Path(vertices, codes)
    
    # Plot the prediction onset times
    plt.scatter(prediction, [1] * len(prediction), 
                marker=custom_marker, color='blue', s=400, label='Prediction')

    # Add labels above each prediction
    if raw_prediction_labels is not None:
        for x, y, label in zip(prediction, [1] * len(prediction), raw_prediction_labels):
            plt.text(x, y - 0.05, f'{label:.2f}', ha='center', fontsize=8, color='red')
    
    for x, y, label in zip(prediction, [1] * len(prediction), prediction):
        plt.text(x, y - 0.1, f'{label:.1f}', ha='center', fontsize=9, color='black')
        
    for i, x in enumerate(prediction):
        plt.text(x, 0.85, f'{i}', ha='center', fontsize=9, color='blue')
    
    # Plot the ground truth onset times
    print("gt len:", len(ground_truth))
    print("pred len:", len(prediction))
    if ground_truth is not None:
        plt.scatter(ground_truth, [0] * len(ground_truth), 
                    marker=custom_marker, color='green', s=400, label='Ground Truth')
        
        # plot lines between prediction and ground truth using alignment
        if alignment is not None:
            for i, j in alignment:
                plt.plot([ground_truth[i], prediction[j]], [0, 1], color='silver', linestyle='--', alpha=0.5)
        
        for x, y, label in zip(ground_truth, [0] * len(ground_truth), ground_truth):
            plt.text(x, y + 0.1, f'{label:.1f}', ha='center', fontsize=9, color='black')
        if raw_score_onset_lengths is not None:
            for x, y, label in zip(ground_truth, [0] * len(ground_truth), raw_score_onset_lengths):
                plt.text(x, y + 0.05, f'{label:.1f}', ha='center', fontsize=9, color='red')
        # print the index for each onset
        for i, x in enumerate(ground_truth):
            plt.text(x, 0.15, f'{i}', ha='center', fontsize=9, color='green')
    
    

    # Customize the plot
    if ground_truth is None:
        plt.yticks([1], ['Prediction'])
        plt.title('Onset Times: Prediction with Labels')
        
    else:
        plt.yticks([1, 0], ['Prediction', 'Ground Truth'])
        plt.title('Onset Times: Ground Truth vs Prediction with Labels')
    plt.xlabel('Onset Times')
    plt.legend()
    os.makedirs(save_path.rsplit('/', 1)[0], exist_ok=True)
    plt.savefig(save_path)

def scale_onsets(onset_lengths, segments, alphas, verbose=False): # Depracated as we're no longer using a piece-wise fit
    """
    Scales onset lengths based on corresponding alphas in segments
    """
    start = 0
    split_onset_lengths = []
    result = onset_lengths.copy()
    
    for segment, alpha in zip(segments, alphas):
        result[start:segment] = np.round(onset_lengths[start:segment] * alpha)
        split_onset_lengths.append(result[start:segment].copy())
        start = segment
    result = result.astype(int)
    # print("Result before gcd division:", result)
    result = np.round(result/(math.gcd(*result) + 1e-8))
    
    return result, split_onset_lengths

def evaluate_onset_trascription(performance_onset_lengths, score_onset_lengths, dtw_func, num_samples=3):
    """
    takes in performance_onset_lengths and label (np arrays of shape (n)) and num samples to determine ratio, and returns average distance, and scaled label
    also takes in a dtw function to generate alignment
    """
    ratios = []
    i = 0
    while len(ratios) < num_samples * 2:
        if not (performance_onset_lengths[i] == 0 or score_onset_lengths[i] == 0):
            ratios.append(performance_onset_lengths[i]/score_onset_lengths[i])
        i += 1
    
    # ratio is median of ratios
    ratio = np.median(ratios)
    # print(ratio)
    if ratio == 0 or ratio is None:
        print("Ratio is 0 or None")
        raise ValueError("Ratio is 0 or None")
        return float("inf"), None, None, None, None
    # input()
    
    scaled_score_onset_lengths = score_onset_lengths * ratio
    scaled_score_onset_times = np.array([0] + list(np.cumsum(scaled_score_onset_lengths)))
    performance_onset_times = np.array([0] + list(np.cumsum(performance_onset_lengths)))
    

    alignment, error = dtw_func(scaled_score_onset_lengths, performance_onset_lengths)
    
    # manually compute squared diff error using alignment
    # squared_diff_error = 0
    
    error = {
        "alignment_error": error, 
    }
    
    return error, performance_onset_times, scaled_score_onset_times, scaled_score_onset_lengths, alignment

def regularization(x, gamma=1, l=2):
    """
    x: np array of shape (n)
    """
    return gamma * sum(x ** l)

def sin_loss(alpha, x, gamma=0, eta=1, debug=False):
    """
    sin loss with asymptote at 0
    x: np array of shape (n)
    """
    x = x.copy() * alpha[0]
    if debug:
        print("scaled:", x)
    
    lesser_than_one = x < 1
    x[lesser_than_one] = (((1 / (x + 1e-6))[lesser_than_one] - 1) ** 2) * eta
    # x[lesser_than_one] = inv(x[lesser_than_one])
    if debug:
        print("less than one:", x)
    x[np.invert(lesser_than_one)] = np.sin(np.pi * x[np.invert(lesser_than_one)]) ** 2
    if debug:
        print("greater than one:", x)
    val = x.mean() + gamma * alpha
    if debug:
        print("val:", val, gamma)
    return val

# def sin_loss_asymp_0_5(alpha, x, gamma=0, eta=1, debug=False):
#     """
#     sin loss with asymptote at 0.5
#     x: np array of shape (n)
#     """
#     def inv(y):
#         return (y - 1) ** 2 / (y - 0.5)
    
#     x = x.copy() * alpha[0]
#     if debug:
#         print("scaled:", x)
        
#     lesser_than_one = x < 1
#     x[lesser_than_one] = inv(x[lesser_than_one]) * eta
    
#     if debug:
#         print("less than one:", x)
#     x[np.invert(lesser_than_one)] = np.sin(np.pi * x[np.invert(lesser_than_one)]) ** 2 
    
#     if debug:
#         print("greater than one:", x)
        
#     val = x.mean() + gamma * np.abs(alpha)
    

def round_loss(alpha, x, gamma=0, debug=False):
    """
    x: np array of shape (n)
    """
    x = x.copy() * alpha[0]
    lesser_than_one = x < 1
    x[lesser_than_one] = ((1/(x+1e-6))[lesser_than_one] - 1) ** 2
    if debug:
        print("less than one:", x)
    x[np.invert(lesser_than_one)] = (x[np.invert(lesser_than_one)] - np.round(x[np.invert(lesser_than_one)])) ** 2
    if debug:
        print("greater than one:", x)
    val = x.mean() + gamma * alpha
    if debug:
        print("val:", val, gamma)
    return val

if __name__ == '__main__':
    # Example usage
    ground_truth = np.array([1, 2, 4, 6])
    prediction = np.array([1.5, 3, 4.5, 6])
    prediction_labels = np.array([0.9, 1.1, 0.8, 1.0])  # Labels for each prediction
    plot_onset_times(prediction, prediction_labels, ground_truth, "test/test.png")