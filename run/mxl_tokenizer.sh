input_path=$1
output_dir=$2
processed_data_dir=$3

echo "mxl tokenization"

python mxl_tokenizer.py \
    --input_path $input_path \
    --output_dir $output_dir \
    --processed_data_dir $processed_data_dir \
