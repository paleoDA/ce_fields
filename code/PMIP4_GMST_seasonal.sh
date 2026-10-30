#!/bin/bash
#SBATCH --job-name=pmip4_gmst_seas
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=PMIP4_GMST_seasonal_%j.out
#SBATCH --error=PMIP4_GMST_seasonal_%j.err

module purge
eval "$(conda shell.bash hook)"
conda activate linkedearth

# Submit from the ce_simul repo root: sbatch code/PMIP4_GMST_seasonal.sh
# (under sbatch, $0 points to Slurm's spool copy, so cd to the submit dir instead)
cd "${SLURM_SUBMIT_DIR:-.}"
python code/PMIP4_GMST_seasonal.py
