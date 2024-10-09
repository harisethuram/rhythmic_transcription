python pretrain.py \
    --processed_data_dir "processed_data/bach-370-chorales"\
    --output_dir "models/test"\
    --embed_size 32 \
    --hidden_size 64 \
    --num_layers 2 \
    --num_epochs 10 \
    --batch_size 32 \
    --learning_rate 0.001 \