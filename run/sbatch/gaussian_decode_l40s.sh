#!/bin/bash
#SBATCH --job-name=gaussDcde
#SBATCH --partition=gpu-l40s
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hsethu@uw.edu
#SBATCH --output=/mmfs1/gscratch/ark/hari/rhythmic_trascription/s_out/gaussian_decode_new_%j.out

export PATH="$PATH:/mmfs1/gscratch/ark/hari/rhythmic_trascription/lilypond-2.25.30/bin"

beam_width=$1
lambda=$2
sigma=$3

source activate rhythm

python run_all/gaussian_decode.py \
    --model_path "output/presentation_results/models/no_barlines/lr_1e-3/b_size_32/emb_64/hid_256/model.pth" \
    --processed_data_dir "processed_data/all/no_barlines" \
    --tempos "60,70,80,90,100,110,120,140,160,180,200" \
    --beam_width "$beam_width" \
    --lambda_param "$lambda" \
    --sigma "$sigma" \
    --root_result_dir "output/gaussian_decoder/hyperparam_search2/beam_width_${beam_width}/lambda_${lambda}/sigma_${sigma}"
    