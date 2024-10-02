import numpy as np

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