prediction_midi_path=$1
gt_midi_path=$2
part_id=$3
output_dir=$4

echo "alignment"

python generate_alignment_given_midi.py \
    --prediction_midi_path "$prediction_midi_path" \
    --ground_truth_midi_path "$gt_midi_path" \
    --part_id $part_id \
    --output_dir "$output_dir"
