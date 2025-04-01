score_path=$1
score_part_id=$2
transcription_path=$3
transcription_part_id=$4
alignment_path=$5
output_dir=$6
processed_data_dir=$7

echo "eval"

python decode_eval.py \
    --score_path $score_path\
    --score_part_id $score_part_id\
    --transcription_path $transcription_path\
    --transcription_part_id $transcription_part_id\
    --alignment_path $alignment_path\
    --output_dir $output_dir\
    --processed_data_dir $processed_data_dir\