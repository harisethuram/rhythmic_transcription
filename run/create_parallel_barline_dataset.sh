input_dir="processed_data/all/barlines"
output_dir="processed_data/all/parallel_barlines"

python create_parallel_barline_dataset.py \
    --input_dir $input_dir \
    --output_dir $output_dir \