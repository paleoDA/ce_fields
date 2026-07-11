#!/bin/bash
#SBATCH --job-name=simul_metadata
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=extract_simul_metadata_%j.out
#SBATCH --error=extract_simul_metadata_%j.err

module purge
eval "$(conda shell.bash hook)"
conda activate linkedearth

cd "$(dirname "$0")/.."            # run from the repo root
python code/extract_simul_metadata.py
