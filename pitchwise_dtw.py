# get alignment between pitch_1 and pitch_2 by DTW using pitch as loss. Can be many-many

import numpy as np
from typing import List, Tuple
from math import inf
from music21 import pitch as m21pitch
import argparse
from fastdtw import fastdtw
import pandas as pd

from src.preprocess.ingestion_utils import read_note_text_file, get_score_note_lengths, midi_part_pitches


import matplotlib.pyplot as plt
import numpy as np


def _rescale(xs, x_min=0.0, x_max=100.0):
    xs = np.asarray(xs, dtype=float)
    if len(xs) == 0:
        return xs
    lo, hi = xs.min(), xs.max()
    if hi == lo:
        # all the same -> put them in the middle
        return np.full_like(xs, (x_min + x_max) / 2.0)
    return x_min + (xs - lo) * (x_max - x_min) / (hi - lo)


def plot_alignment_staggered(seq1, seq2, alignment, path):
    """
    Plot a DTW alignment with staggered x-coordinates.

    seq1, seq2: lists of pitches (or None for rests).
    alignment: list of (i, j) in path order; i/j can be None for gaps.

    Logic:
      - Each alignment step k gets an x_k linearly from 0..100 over the path.
      - For each note index i in seq1, its x is the average of x_k for all
        steps where alignment[k][0] == i. Same for seq2 / j.
      - Then, for each sequence separately, its note x's are rescaled so that
        its first and last note are at 0 and 100.
    """
    n1, n2 = len(seq1), len(seq2)
    L = len(alignment)

    if L == 0:
        raise ValueError("Alignment is empty")

    # 1) x-position per alignment step (path index)
    if L == 1:
        step_x = np.array([50.0])
    else:
        step_x = np.linspace(0.0, 100.0, L)

    # 2) collect x's per note index
    xs1_raw = [[] for _ in range(n1)]
    xs2_raw = [[] for _ in range(n2)]

    for k, (i, j) in enumerate(alignment):
        if i is not None:
            xs1_raw[i].append(step_x[k])
        if j is not None:
            xs2_raw[j].append(step_x[k])

    # 3) average per note; if a note somehow never appears, fall back to index-based
    x1_raw = np.zeros(n1)
    for i in range(n1):
        if xs1_raw[i]:
            x1_raw[i] = np.mean(xs1_raw[i])
        else:
            # fallback: linear in index
            x1_raw[i] = i

    x2_raw = np.zeros(n2)
    for j in range(n2):
        if xs2_raw[j]:
            x2_raw[j] = np.mean(xs2_raw[j])
        else:
            x2_raw[j] = j

    # 4) rescale per sequence so first note = 0, last = 100
    x1 = _rescale(x1_raw, 0.0, 100.0) if n1 > 0 else []
    x2 = _rescale(x2_raw, 0.0, 100.0) if n2 > 0 else []

    # 5) plotting
    fig, ax = plt.subplots(figsize=(100, 4))

    # baselines
    ax.hlines(1, 0, 100, linestyle="--", linewidth=1)
    ax.hlines(0, 0, 100, linestyle="--", linewidth=1)

    # seq1
    ax.scatter(x1, [1] * n1, zorder=3)
    for i, (x, pitch) in enumerate(zip(x1, seq1)):
        label = "None" if pitch is None else str(pitch)
        ax.text(x, 1 + 0.04, label,
                ha="center", va="bottom", rotation=45, fontsize=8)

    # seq2
    ax.scatter(x2, [0] * n2, zorder=3)
    for j, (x, pitch) in enumerate(zip(x2, seq2)):
        label = "None" if pitch is None else str(pitch)
        ax.text(x, 0 - 0.04, label,
                ha="center", va="top", rotation=45, fontsize=8)

    # alignment lines
    for i, j in alignment:
        if i is None or j is None:
            continue
        ax.plot([x1[i], x2[j]], [1, 0], linewidth=0.8, alpha=0.7, zorder=2)

    ax.set_xlim(-5, 105)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["seq2", "seq1"])
    ax.set_xlabel("Normalized position (0 → 100 along path)")
    ax.set_title("Pitch Alignment (staggered many-to-one / one-to-many)")

    plt.tight_layout()
    plt.savefig(path)
    return fig, ax


def get_dtw_alignment_pitchwise(pitch_1: List[str], pitch_2: List[str]) -> Tuple[float, List[List[int]]]:
    """
    Get DTW alignment between two pitch sequences using pitch as loss
    pitch_1: list of pitches (str) for sequence 1
    pitch_2: list of pitches (str) for sequence 2
    returns: alignment as list of [i, j] pairs where i is index in pitch_1 and j is index in pitch_2
    """
    want_rests = False
    if want_rests:
        pitch_1 = [m21pitch.Pitch(p).midi if p is not None else p for p in pitch_1]
        pitch_2 = [m21pitch.Pitch(p).midi if p is not None else p for p in pitch_2]
    else:
        pitch_1 = [m21pitch.Pitch(p).midi for p in pitch_1 if p is not None]
        pitch_2 = [m21pitch.Pitch(p).midi for p in pitch_2 if p is not None]
    
    def cost(p1, p2):
        # binary cost: 0 if same pitch, 1 otherwise
        return 0 if p1 == p2 else 1
    
    deletion_cost = 0.2
    
    D = np.zeros((len(pitch_1)+1, len(pitch_2)+1))
    D[0, 1:] = np.cumsum([deletion_cost] * len(pitch_2))
    D[1:, 0] = np.cumsum([deletion_cost] * len(pitch_1))
    D[0, 0] = 0
    
    move_labels = ["match", "delete_seq1", "delete_seq2"]
    choice_matrix = np.empty((len(pitch_1)+1, len(pitch_2)+1), dtype=object)
    
    choice_matrix[1:, 0] = "delete_seq1" # first column is deletions from seq2, as seq1 notes align to nothing
    choice_matrix[0, 1:] = "delete_seq2" # first row is deletions from seq1, as seq2 notes align to nothing
    choice_matrix[0, 0] = None  # starting point, no move
    
    
    
    for i in range(1, len(pitch_1)+1):
        for j in range(1, len(pitch_2)+1):
            choices = [
                D[i-1, j-1] + cost(pitch_1[i-1], pitch_2[j-1]),  # match/mismatch
                D[i-1, j] + deletion_cost,                       # deletion from seq1
                D[i, j-1] + deletion_cost                        # deletion from seq2
            ]
            D[i, j] = min(choices)
            choice_matrix[i, j] = move_labels[np.argmin(choices)]
            
    alignment = []
    i, j = len(pitch_1), len(pitch_2)

    while i > 0 or j > 0:
        move = choice_matrix[i, j]

        if move == "match":
            # record a true alignment between i-1 and j-1
            alignment.append([i - 1, j - 1])
            i -= 1
            j -= 1

        elif move == "delete_seq1":
            # seq1[i-1] is unmatched (deleted); don't add to alignment
            i -= 1

        elif move == "delete_seq2":
            # seq2[j-1] is unmatched (deleted); don't add to alignment
            j -= 1

        else:
            raise RuntimeError(f"Unexpected move {move} at ({i}, {j})")

    alignment.reverse()
    
    return D[-1, -1], alignment, pitch_1, pitch_2
            
    
    # distance, path = fastdtw(pitch_1, pitch_2, dist=cost)
    # return distance, path, [m21pitch.Pitch(p) for p in pitch_1], [m21pitch.Pitch(p) for p in pitch_2]
    
    
if __name__ == "__main__":
    metadata = pd.read_csv("metadata/URMP/metadata.csv")
    for idx, row in metadata.iterrows():
        perf_path = row["notes_path"]
        gt_path = row["score_path"]
        output_path = row["alignment_path"]
        part_id = int(row["part_id"]) - 1
        print(idx, row["piece_name"], row["part_id"])
        
        _, perf_pitches, _ = read_note_text_file(perf_path)
        gt_pitches = midi_part_pitches(gt_path.replace("xml", "mid"), part_id=part_id)
        _, alignment, gt_pitches, perf_pitches = get_dtw_alignment_pitchwise(gt_pitches, perf_pitches)
        print(alignment)
        input()
        plot_alignment_staggered(gt_pitches, perf_pitches, alignment, output_path.replace(".json", ".png"))