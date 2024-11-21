input_path=$1
output_dir=$2
score_path=$3
score_part_number=$4
base_value=$5

echo "input_path: $input_path"
echo "output_dir: $output_dir"
echo "score_path: $score_path"
echo "score_part_number: $score_part_number"
echo "base_value: $base_value"

python beta_transcription.py \
    --input_path $input_path \
    --output_dir $output_dir \
    --eval \
    --score_path $score_path \
    --score_part_number $score_part_number \
    --base_value $base_value \
