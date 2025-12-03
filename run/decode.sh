note_info_path=$1
output_dir=$2
base_value=$3
model_path=$4
want_mixing=$5
processed_data_dir=$6
# score_path=$4
# part_id=$5
# alignment_path=$6
echo $1 $2 $3
echo $4
echo $5

# python decode.py \
#     --language_model_path "models/bach_fugues/pretrain_hparam_search/lr_1e-3/b_size_64/emb_64/hid_256/model.pth" \
#     --channel_model_path "output/test_beta/beta_channel.pth" \
#     --processed_data_dir "processed_data/bach_fugues" \
#     --note_info_path "$note_info_path" \
#     --decode_method "$method" \
#     --output_dir "$output_dir" \
#     --base_value "$base_value" 

python decode.py \
    --language_model_path "$model_path" \
    --channel_model_path "models/old/tmp/test_beta/beta_channel.pth" \
    --processed_data_dir "$processed_data_dir" \
    --note_info_path "$note_info_path" \
    --decode_method "beam_search" \
    --output_dir "$output_dir" \
    --base_value "$base_value" \
    --want_mixing "$want_mixing" \
    # --eval \
    # --score_path "$score_path" \
    # --score_part_id $part_id \
    # --alignment_path "$alignment_path" \