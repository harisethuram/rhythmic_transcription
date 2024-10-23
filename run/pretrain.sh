lrs=(1e-3) # 1e-4 1e-5)
batch_sizes=(32) # 64 128)
embed_sizes=(32) # 64 128)
hidden_sizes=(64) # 128 256)

for lr in "${lrs[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        for embed_size in "${embed_sizes[@]}"; do
            for hidden_size in "${hidden_sizes[@]}"; do
                echo "Running pretrain with lr=$lr, batch_size=$batch_size, embed_size=$embed_size, hidden_size=$hidden_size"
                python pretrain.py \
                    --processed_data_dir "processed_data/bach_fugues"\
                    --output_dir "results/bach_fugues_new_tokenizer/pretrain_hparam_search/lr_$lr/b_size_$batch_size/emb_$embed_size/hid_$hidden_size"\
                    --embed_size $embed_size \
                    --hidden_size $hidden_size \
                    --num_layers 2 \
                    --num_epochs 1000 \
                    --batch_size $batch_size \
                    --learning_rate $lr 
            done
        done
    done
done

# python pretrain.py \
#     --processed_data_dir "processed_data/bach-370-chorales"\
#     --output_dir "models/test"\
#     --embed_size 32 \
#     --hidden_size 64 \
#     --num_layers 2 \
#     --num_epochs 200 \
#     --batch_size 32 \
#     --learning_rate 0.001 \