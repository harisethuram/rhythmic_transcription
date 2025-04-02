# Takes in path for the transcribed and ground-truth midi files and finds the optimal alignment. Saves the alignment as json file.
# Currently used for aligning the musescore and ground truth for evaluation. 
import argparse
import numpy as np
import os
import json

from src.preprocess.ingestion_utils import midi_to_onsets
from src.eval.utils import evaluate_onset_trascription, plot_onset_times
from src.eval.dtw import hybrid_DTW


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_midi_path", type=str, required=True, help="Path to the transcribed MIDI file")
    parser.add_argument("--ground_truth_midi_path", type=str, required=True, help="Path to the ground truth MIDI file")
    parser.add_argument("--part_id", type=int, required=True, help="Part ID in ground truth")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save json alignment file")
    args = parser.parse_args()
    print(args)
    print("loading files...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    prediction_onsets = midi_to_onsets(args.prediction_midi_path)[0]
    ground_truth_onsets = midi_to_onsets(args.ground_truth_midi_path)[args.part_id-1]
    
    prediction_onset_lengths = np.diff(prediction_onsets)
    ground_truth_onset_lengths = np.diff(ground_truth_onsets)
    
    print("getting alignment...")
    error, scaled_prediction_onsets, scaled_ground_truth_onsets, _, alignment = evaluate_onset_trascription(prediction_onset_lengths, ground_truth_onset_lengths, hybrid_DTW)

    print("saving and plotting...")
    with open(os.path.join(args.output_dir, "alignment.json"), "w") as f:
        json.dump(alignment, f)
    
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(
            {
                "error": error,
                "scaled_prediciton_onsets": [str(s) for s in scaled_prediction_onsets.tolist()], 
                "scaled_ground_truth_onsets": [str(s) for s in scaled_ground_truth_onsets.tolist()], 
                "prediction_onsets": [str(s) for s in prediction_onsets.tolist()],
                "ground_truth_onsets": [str(s) for s in ground_truth_onsets.tolist()]
            }, 
            f
        )
        
    plt_path = os.path.join(args.output_dir, "alignment.png")
    plot_onset_times(prediction=scaled_prediction_onsets, ground_truth=scaled_ground_truth_onsets, alignment=alignment, save_path=plt_path)