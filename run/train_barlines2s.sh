#!/bin/bash
#SBATCH --job-name=barline_s2s_train
#SBATCH --partition=gpu-l40
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hsethu@uw.edu
#SBATCH --output=/mmfs1/gscratch/ark/hari/rhythmic_trascription/out/%j.out


data_dir=$1
output_dir=$2
hidden_dim=$3
num_layers=$4
batch_size=$5
lr=$6

echo "Data Directory: $data_dir"
echo "Output Directory: $output_dir"
echo "Hyperparameters: hidden_dim=$hidden_dim, num_layers=$num_layers, batch_size=$batch_size, learning_rate=$lr"

python train_barline_s2s.py \
    --data_dir $data_dir \
    --output_dir $output_dir \
    --hidden_dim $hidden_dim \
    --num_layers $num_layers \
    --batch_size $batch_size \
    --learning_rate $lr \