echo $1
echo $2
echo $3
echo $4
echo $5

python add_barlines.py \
    --model_path "$4" \
    --input_path "$1" \
    --data_dir "$5" \
    --output_path "$2" \
    --note_info_path "$3" \