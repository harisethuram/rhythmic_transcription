#!/bin/bash
#SBATCH --job-name=mix_decode
#SBATCH --partition=gpu-l40s
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hsethu@uw.edu
#SBATCH --output=/mmfs1/gscratch/ark/hari/rhythmic_trascription/out/pretrain_decode%j.out

echo "Running pretrain decode script"
source activate rhythm
bash run/complex/pretrain_decode/pretrain_decode.sh
