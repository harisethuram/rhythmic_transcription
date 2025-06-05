#!/bin/bash
#SBATCH --job-name=add_barlines
#SBATCH --partition=gpu-l40
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hsethu@uw.edu
#SBATCH --output=/mmfs1/gscratch/ark/hari/rhythmic_trascription/out/add_barlines.out

python run_all/add_barlines.py