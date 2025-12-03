# pretrain three models
# no barlines
no_barlines_dir="output/presentation_results/models/elucidation_experiment/no_barlines"
# bash run/complex/pretrain_decode/pretrain.sh "output/presentation_results/models/elucidation_experiment/no_barlines" "processed_data/elucidation_experiment/no_barlines"

# # barlines
# bash run/complex/pretrain_decode/pretrain.sh "output/presentation_results/models/elucidation_experiment/barlines" "processed_data/elucidation_experiment/barlines"

# # barlines and measure lengths
# bash run/complex/pretrain_decode/pretrain.sh "output/presentation_results/models/elucidation_experiment/barlines_and_measure_lengths" "processed_data/elucidation_experiment/barlines_and_measure_lengths"
# 
# # now, decode using no barlines model
python run_all/decode.py \
    --model_path $no_barlines_dir/model.pth \
    --root_result_dir "output/presentation_results/decode/elucidation_experiment/no_barlines/{piece}/{base_value}" \
    --processed_data_dir "processed_data/elucidation_experiment/no_barlines" \
    --want_mixing