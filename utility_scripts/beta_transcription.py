import argparse
import json
from scipy.stats import beta
import os
import numpy as np
from music21 import converter, note, stream, duration
from fractions import Fraction
from tslearn.metrics import dtw_path
import matplotlib.pyplot as plt

from src.model.model_utils import get_beta_params_from_mode_and_spread
from src.preprocess.ingestion_utils import get_score_note_lengths

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert onset tuples of format (onset_length, note portion, rest portion, note name) to a score without barlines.")
    parser.add_argument("--input_path", type=str, help="Path to the input json file.")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory.")
    parser.add_argument("--base_value", type=float, default=1/16, help="What 1 in the onset lenghts corresponds to (16 corresponds to 16th note).")
    parser.add_argument("--eval", action="store_true", help="If true, the script will evaluate the transcription.")
    parser.add_argument("--score_path", type=str, default=None, help="File name of the score onset times, only required if eval is true")
    parser.add_argument("--score_part_number", type=int, default=None, help="Part number of the score, only required if eval is true")
    args = parser.parse_args()
    
    print(args)
    
    with open(args.input_path, "r") as f:
        data = np.array(json.load(f))
    base_value = args.base_value * 4
    
    # modes for the beta distribution
    modes = [1/4, 1/3, 1/2, 2/3, 3/4, 1]
    s = 0.1
    alphas = []
    betas = []
    for mode in modes:
        alpha, beta_param = get_beta_params_from_mode_and_spread(mode, s)
        alphas.append(alpha)
        betas.append(beta_param)
    
    note_lengths = []
    rest_lengths = []
    note_names = []
    for row in data:
        onset_length = float(row[0])
        note_portion = float(row[1])
        rest_portion = float(row[2])
        note_names.append(row[3])
        if note_portion == 0:
            note_lengths.append(0)
        else:
            idx = np.argmax([beta.pdf(note_portion, alphas[i], betas[i]) for i in range(len(modes))])
            note_lengths.append(onset_length * modes[idx] * base_value)
            rest_lengths.append(onset_length * (1 - modes[idx]) * base_value)
    
    notes_and_rests = [[note_length, rest_length] for note_length, rest_length in zip(note_lengths, rest_lengths)]
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, "notes_and_rests.json"), "w") as f:
        json.dump(notes_and_rests, f)
    
    
    # Write score as MusicXML
    score = stream.Score()
    part = stream.Part()
    score.append(part)    
    
    # Iterate over both lists
    for idx, (n_length, r_length) in enumerate(zip(note_lengths, rest_lengths)):
        # Create and append a note
        if n_length > 1e-3:
            n = note.Note(note_names[idx])
            n.duration = duration.Duration(n_length)
            part.append(n)
        
        # Only add a rest if its length is greater than 0
        if r_length > 1e-3:
            r = note.Rest()
            r.duration = duration.Duration(r_length)
            part.append(r)
                   
    score.write('musicxml', fp=os.path.join(args.output_dir, "score.musicxml"))
    
    # plot the transcription
    total_lengths = np.array([0] + note_lengths) + np.array([0] + rest_lengths)
    onset_times = np.cumsum(total_lengths)
    
    # scale note lengths by 10 to make them visible
    scale_factor = 1
    plot_note_lengths = np.array(note_lengths) * scale_factor
    plot_rest_lengths = np.array(rest_lengths) * scale_factor
    plot_onset_times = onset_times * scale_factor
    
    # make plot wide
    fig_length = min(plot_onset_times[-1], 655)
    print("figure size:", fig_length, 4)
    # input()
    plt.figure(figsize=(fig_length, 4))
    for i, (note_length, rest_length) in enumerate(zip(plot_note_lengths, plot_rest_lengths)):
        plt.hlines(y=0, xmin=plot_onset_times[i], xmax=plot_onset_times[i] + note_length, colors='dodgerblue', linewidth=2)
        plt.plot(plot_onset_times[i], 0, marker='o', markersize=5, color='dodgerblue', label='Pred notes' if i == 0 else "")
        
        if rest_length != 0:
            plt.hlines(y=0, xmin=plot_onset_times[i] + note_length, xmax=plot_onset_times[i] + note_length + rest_length, colors='coral', linewidth=2)
            plt.plot(plot_onset_times[i] + note_length, 0, marker='o', markersize=5, color='coral', label='Pred rests' if i == 0 else "")
    
    # add y tick at 0 for prediction
    
    
    
    if args.eval:
        score_info, _ = get_score_note_lengths(args.score_path, args.score_part_number)
        
        score_note_lengths = score_info[:,0]
        score_rest_lengths = score_info[:,1]
        
        # adjust scales to ensure they're the same
        
        # ratios = []
        # i = 0
        
        # while len(ratios) < 3:
        #     if not (note_lengths[i] == 0 or note_lengths[i] == 0):
        #         ratios.append(note_lengths[i]/score_note_lengths[i])
        #     i += 1
        
        # # ratio is median of ratios
        # ratio = np.median(ratios)
        
        # score_note_lengths = score_note_lengths * ratio
        # score_rest_lengths = score_rest_lengths * ratio
        
        note_alignment = dtw_path(note_lengths, score_note_lengths)[0]
        rest_alignment = dtw_path(rest_lengths, score_rest_lengths)[0]
        
        note_error = sum([(note_lengths[i] - score_note_lengths[j])**2 for i, j in note_alignment])/len(note_lengths)
        rest_error = sum([(rest_lengths[i] - score_rest_lengths[j])**2 for i, j in rest_alignment])/len(rest_lengths)
        
        binary_note_accuracy = sum([1 for i, j in note_alignment if note_lengths[i] == score_note_lengths[j]])/len(note_lengths)
        binary_rest_accuracy = sum([1 for i, j in rest_alignment if rest_lengths[i] == score_rest_lengths[j]])/len(rest_lengths)
        
        all_results = {
            "note_error": note_error,
            "rest_error": rest_error,
            "binary_note_accuracy": binary_note_accuracy,
            "binary_rest_accuracy": binary_rest_accuracy
        }
        
        all_results_path = os.path.join(args.output_dir, "results.json")
        with open(all_results_path, "w") as f:
            json.dump(all_results, f)
        print("Note error:", note_error)
        print("Rest error:", rest_error)
        print("Binary note accuracy:", binary_note_accuracy)
        print("Binary rest accuracy:", binary_rest_accuracy)
        print("Total error:", note_error + rest_error)
        
        # plot the ground truth
        total_score_lengths = np.array([0] + score_note_lengths.tolist()) + np.array([0] + score_rest_lengths.tolist())
        score_onset_times = np.cumsum(total_score_lengths)
        
        for i, (note_length, rest_length) in enumerate(zip(score_note_lengths, score_rest_lengths)):
            plt.hlines(y=-0.1, xmin=score_onset_times[i], xmax=score_onset_times[i] + note_length, colors='deepskyblue', linewidth=2)
            plt.plot(score_onset_times[i], -0.1, marker='o', markersize=5, color='deepskyblue', label='GT notes' if i == 0 else "")
            
            
            if rest_length != 0:
                plt.hlines(y=-0.1, xmin=score_onset_times[i] + note_length, xmax=score_onset_times[i] + note_length + rest_length, colors='red', linewidth=2)
                plt.plot(score_onset_times[i] + note_length, -0.1, marker='o', markersize=5, color='red', label='GT rests' if i == 0 else "")
            
        # plot lines connecting the ground truth and the prediction using the alignment
        for i, j in note_alignment:
            
            plt.plot([plot_onset_times[i], score_onset_times[j]], [0, -0.1], color='black', linestyle='--')
            if rest_lengths[i] != 0:
                pred_rest_start_time = plot_onset_times[i] + plot_note_lengths[i]
                score_rest_start_time = score_onset_times[j] + score_note_lengths[j]
                plt.plot([pred_rest_start_time, score_rest_start_time], [0, -0.1], color='grey', linestyle='--')
            # plt.plot([plot_onset_times[i], score_onset_times[j]], [0, -0.1], color='black')
            
    
    plt_path = os.path.join(args.output_dir, "transcription.png")
    plt.xlabel('Time')
    
    if not args.eval:
        plt.yticks([0], ['Prediction'])
        
    else:
        plt.yticks([0, -0.1], ['Prediction', 'Ground Truth'])

    
    plt.title('Transcription')
    plt.legend()
    plt.savefig(plt_path)
    
    # input("Press Enter to continue...")
        
        