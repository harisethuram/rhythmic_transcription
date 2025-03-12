note_info_path=$1
output_dir=$2
base_value=$3
method=$4


python decode.py \
    --language_model_path "models/bach_fugues/pretrain_hparam_search/lr_1e-3/b_size_64/emb_64/hid_256/model.pth" \
    --channel_model_path "output/test_beta/beta_channel.pth" \
    --processed_data_dir "processed_data/bach_fugues" \
    --note_info_path "$note_info_path" \
    --decode_method "$method" \
    --output_dir "$output_dir" \
    --base_value "$base_value" 