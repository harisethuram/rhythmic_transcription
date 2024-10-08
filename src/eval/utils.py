import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import os

def plot_onset_times(prediction, prediction_labels, ground_truth=None, save_path=None):
    # Create a new figure
    plt.figure(figsize=(9, 4))
    t = 1
    vertices = [
        (0, 0), (t*2, t), (0, 2*t),  # Define a triangle-like shape
        (0, 0)  # Close the path
    ]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    custom_marker = Path(vertices, codes)
    
    # Plot the ground truth onset times
    if ground_truth is not None:
        plt.scatter(ground_truth, [1] * len(ground_truth), 
                    marker=custom_marker, color='green', s=200, label='Ground Truth')
    
    # Plot the prediction onset times
    plt.scatter(prediction, [0] * len(prediction), 
                marker=custom_marker, color='blue', s=200, label='Prediction')

    # Add labels above each prediction
    for x, y, label in zip(prediction, [0] * len(prediction), prediction_labels):
        plt.text(x, y + 0.1, f'{label:.2f}', ha='center', fontsize=10, color='red')

    # Customize the plot
    if ground_truth is None:
        plt.yticks([0], ['Prediction'])
        plt.title('Onset Times: Prediction with Labels')
        
    else:
        plt.yticks([0, 1], ['Prediction', 'Ground Truth'])
        plt.title('Onset Times: Ground Truth vs Prediction with Labels')
    plt.xlabel('Onset Times')
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    os.makedirs(save_path.rsplit('/', 1)[0], exist_ok=True)
    plt.savefig(save_path)

def scale_onsets(onset_lengths, segments, alphas):
    start = 0
    split_onset_lengths = []
    result = onset_lengths.copy()
    
    for segment, alpha in zip(segments, alphas):
        result[start:segment] = np.round(onset_lengths[start:segment] * alpha)
        split_onset_lengths.append(result[start:segment].copy())
        start = segment
    
    return result, split_onset_lengths


def evaluate_onset_trascription(pred, label):
    """
    takes in prediction and label (np arrays of shape (n)) and returns average distance
    """
    ratio = np.min(pred[pred > 0])/np.min(label[label > 0])
    error = np.sum((pred - ratio * label)**2)
    return error, ratio * label

def sin_loss(alpha, x, gamma=0, debug=False):
    """
    x: np array of shape (n)
    """
    x = x.copy() * alpha[0]
    if debug:
        print("scaled:", x)
    
    lesser_than_one = x < 1
    x[lesser_than_one] = ((1/(x+1e-6))[lesser_than_one] - 1) ** 2
    if debug:
        print("less than one:", x)
    x[np.invert(lesser_than_one)] = np.sin(np.pi * x[np.invert(lesser_than_one)]) ** 2
    if debug:
        print("greater than one:", x)
    val = x.mean() + gamma * alpha
    if debug:
        print("val:", val, gamma)
    return val

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