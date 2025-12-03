
source activate rhythm

echo "Running pretrain"
python pretrain.py \
    --processed_data_dir "$2"\
    --output_dir "$1" \
    --embed_size 32 \
    --hidden_size 256 \
    --num_layers 2 \
    --num_epochs 300 \
    --batch_size 32 \
    --learning_rate 1e-3
