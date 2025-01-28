python decode.py \
    --model_path "models/bach_fugues/pretrain_hparam_search/lr_1e-3/b_size_64/emb_64/hid_256/model.pth" \
    --processed_data_dir "processed_data/bach_fugues" \
    --note_info_path "output/test_results/URMP/1_Jupiter_1/note_info.json" \
    --decode_method "greedy" \