#!/bin/bash
#SBATCH --job-name=gaussDcde
#SBATCH --partition=gpu-a40
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hsethu@uw.edu
#SBATCH --output=/mmfs1/gscratch/ark/hari/rhythmic_trascription/s_out/gaussian_decode_%j.out

export PATH="$PATH:/mmfs1/gscratch/ark/hari/rhythmic_trascription/lilypond-2.25.30/bin"

beam_width=$1
lambda=$2
sigma=$3

source activate rhythm

python run_all/gaussian_decode.py \
    --model_path "output/presentation_results/models/no_barlines/lr_1e-3/b_size_32/emb_64/hid_256/model.pth" \
    --processed_data_dir "processed_data/all/no_barlines" \
    --tempos "60,65,70,75,80,85,90,95,100,105,110,115,120,130,140,150,160,170,180,200,220" \
    --beam_width "$beam_width" \
    --lambda_param "$lambda" \
    --sigma "$sigma" \
    --root_result_dir "output/gaussian_decoder/hyperparam_search/beam_width_${beam_width}/lambda_${lambda}_sigma_${sigma}"
    