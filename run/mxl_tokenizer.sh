input_path=$1
output_path=$2

echo "mxl tokenization"

python mxl_tokenizer.py \
    --input_path $input_path \
    --output_path $output_path \
    --processed_data_dir "processed_data/test_all" \
