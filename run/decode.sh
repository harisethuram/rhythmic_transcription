note_info_path=$1
output_dir=$2
base_value=$3
score_path=$4
part_id=$5
alignment_path=$6


# python decode.py \
#     --language_model_path "models/bach_fugues/pretrain_hparam_search/lr_1e-3/b_size_64/emb_64/hid_256/model.pth" \
#     --channel_model_path "output/test_beta/beta_channel.pth" \
#     --processed_data_dir "processed_data/bach_fugues" \
#     --note_info_path "$note_info_path" \
#     --decode_method "$method" \
#     --output_dir "$output_dir" \
#     --base_value "$base_value" 

python decode.py \
    --language_model_path "models/tmp/test_all/model.pth" \
    --channel_model_path "models/tmp/test_beta/beta_channel.pth" \
    --processed_data_dir "processed_data/test_all" \
    --note_info_path "$note_info_path" \
    --decode_method "beam_search" \
    --output_dir "$output_dir" \
    --base_value "$base_value" \
    --eval \
    --score_path "$score_path" \
    --score_part_id $part_id \
    --alignment_path "$alignment_path" \