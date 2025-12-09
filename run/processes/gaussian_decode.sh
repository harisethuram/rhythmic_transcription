echo $1
echo $2
echo $3
echo $4
echo $5
echo $6
echo $7

python gaussian_decode.py \
        --rhythm_lstm_path $1 \
        --processed_data_dir $2 \
        --input_path $3 \
        --tempo $4 \
        --output_dir $5 \
        --beam_width $6 \
        --lambda_param $7 \
        --sigma $8