#!/bin/bash
#SBATCH --job-name=pmip4_check
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=check_pmip4_zarr_%j.out
#SBATCH --error=check_pmip4_zarr_%j.err

module purge
eval "$(conda shell.bash hook)"
conda activate snakemake3          # update if the ce_fields environment has a different name

cd "$(dirname "$0")/.."            # run from the repo root
python scripts/check_pmip4_zarr.py
