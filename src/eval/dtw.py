# computes one to many DTW alignment between two sequences of onset lengths
# Dij = min_k (D(i-1, k-1) + cost(s_i, a_k...a_j))
import numpy as np
from tslearn.metrics import dtw_path
from tqdm import tqdm

def one_to_one_DTW(seq_1, seq_2):
    return dtw_path(seq_1, seq_2)[0]

def cost_function_one_to_one(D, seq_1, seq_2):
    pass

def one_to_many_DTW(seq_1, seq_2):
    """
    Computes the one-to-many DTW alignment between two sequences of onset lengths
    one-to-many from seq_1 to seq_2
    """
    if len(seq_1) > len(seq_2):
        seq_1 = seq_1[:len(seq_2)]
    D = np.zeros((len(seq_1)+1, len(seq_2)+1))
    print(len(seq_1), len(seq_2))
    
    # initialize the first row and column
    D[0, 1:] = np.inf
    D[1:, 0] = np.inf
    D[0, 0] = 0
    
    choices = np.zeros((len(seq_1)+1, len(seq_2)+1))
    
    for i in range(1, len(seq_1)+1):
        for j in range(1, len(seq_2)+1):
            cost_k = [D[i-1, k-1] + abs(sum(seq_2[k-1:j]) - seq_1[i-1]) for k in range(1, j+1)]
            
            D[i, j] = min(cost_k)
            choices[i, j] = np.argmin(cost_k) + 1

    # backtracking
    alignment = []
    i = len(seq_1)
    j = len(seq_2)
    while i > 0:
        k = int(choices[i, j])
        # print(i, j, k, choices[i, j])
        # input()
        for l in range(j, k-1, -1):
            alignment.insert(0, [i-1, l-1])
        i -= 1
        j = k - 1
    
    print(alignment)
    return np.array(alignment).tolist(), D[-1, -1]


def many_to_one_DTW(seq_1, seq_2):
    """
    Computes the many-to-one DTW alignment between two sequences of onset lengths
    many-to-one from seq_1 to seq_2
    """
    alignment, score = one_to_many_DTW(seq_2, seq_1)
    
    # flip first and second columns
    alignment = np.array(alignment)[:, ::-1]
    
    return alignment.tolist(), score
        

def hybrid_DTW(seq_1, seq_2):
    """
    Computes the hybrid DTW alignment between two sequences of onset lengths
    """
    D = np.zeros((len(seq_1)+1, len(seq_2)+1))
    print(len(seq_1), len(seq_2))
    
    # initialize the first row and column
    D[0, 1:] = np.cumsum([s2 ** 2 for s2 in seq_2]) # basically deletions
    D[1:, 0] = np.cumsum([s1 ** 2 for s1 in seq_1]) # basically insertions
    D[0, 0] = 0
    
    choices = np.full((2, len(seq_1)+1, len(seq_2)+1), -1)
    choices[:, 0, 0] = [3, -1]
    choices[0, 0, 1:] = [1] * len(seq_2)
    choices[0, 1:, 0] = [2] * len(seq_1)
    
    
    for i in tqdm(range(1, len(seq_1)+1)):
        for j in range(1, len(seq_2)+1):
            # case 1: deletion
            del_cost = D[i, j-1] + seq_2[j-1] ** 2
            
            # case 2: insertion
            ins_cost = D[i-1, j] + seq_1[i-1] ** 2
            
            # case 3: multiple seq_1 sum up to one seq_2
            cost_k = [D[k, j-1] + (seq_2[j-1] - sum(seq_1[k:i])) ** 2 for k in range(i)]
            min_cost_k = min(cost_k)
            argmin_cost_k = np.argmin(cost_k)
            
            # case 4: multiple seq_2 sum up to one seq_1
            cost_l = [D[i-1, l] + (seq_1[i-1] - sum(seq_2[l:j])) ** 2 for l in range(j)]
            min_cost_l = min(cost_l)
            argmin_cost_l = np.argmin(cost_l)
            
            # print(i, j)
            # print(seq_1[i-1], seq_2[j-1])
            # print(del_cost)
            # print(ins_cost)
            # print(min_cost_k, argmin_cost_k, cost_k)
            # print(min_cost_l, argmin_cost_l, cost_l)
            # input()
            
            
            # choose the minimum cost and the corresponding case and the k or l value if applicable
            min_cost = min(del_cost, ins_cost, min_cost_k, min_cost_l)
            if min_cost == del_cost:
                choices[:, i, j] = [1, -1]
            elif min_cost == ins_cost:
                choices[:, i, j] = [2, -1]
            elif min_cost == min_cost_k:
                choices[:, i, j] = [3, argmin_cost_k]
            else:
                choices[:, i, j] = [4, argmin_cost_l]
    
            D[i, j] = min_cost  
            
    # print(D) 
    # print(choices)
    # backtracking
    alignment = []
    i = len(seq_1)
    j = len(seq_2)
    print("cost:", D[-1, -1])
    print(D)
    
    # everything is done in reference to seq_1 as that's the ground truth
    # return None
    while i > 0 and j > 0:
        choice = int(choices[0, i, j])
        if choice == 1:
            # no alignment as we're deleting this
            j -= 1
        elif choice == 2:
            # no alignment as we're inserting this
            i -= 1
        elif choice == 3:
            k = int(choices[1, i, j])
            curr = []
            for idx in range(k, i):
                curr.append((idx, j-1))
            
            alignment = curr + alignment
            i = k   
            j -= 1
        else:
            l = int(choices[1, i, j])
            curr = []
            for idx in range(l, j):
                curr.append((i-1, idx))
                
            alignment = curr + alignment
            j = l    
            i -= 1
    
    print(alignment)
    return alignment, D[-1, -1]
          


if __name__ == "__main__":
    # seq_1 = np.array([0.5, 0.5, 2, 1, 7])
    # seq_2 = np.array([1, 1, 1, 1, 3, 4])
    # seq_1 = np.array([2, 2, 2,   # sums to 6
    #     3, 5, 2,   # sums to 10
    #     3, 3,      # sums to 6
    #     2, 2, 5,   # sums to 9
    #     1, 3])
    # seq_2 = np.array([6, 10, 6, 9, 4])
    seq_1 = np.array([1, 1, 3, 2, 1, 4, 4, 5, 1, 1, 2, 2, 1])
    seq_2 = np.array([2, 3, 3, 4, 4, 7, 4, 2])
    hybrid_DTW(seq_1, seq_2)