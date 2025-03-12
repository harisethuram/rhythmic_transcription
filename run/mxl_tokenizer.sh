input_path=$1
output_path=$2

echo "input_path: $input_path"
echo "output_path: $output_path"

python mxl_tokenizer.py \
    --input_path $input_path \
    --output_path $output_path \
    --processed_data_dir "processed_data/all" \
