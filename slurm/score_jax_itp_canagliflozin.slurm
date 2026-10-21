#!/bin/bash
# Organ calls for the JAX ITP canagliflozin slides.
# CPU only. A GPU does not help: this is mean-pooling and a linear model.
#
#SBATCH --job-name=cana-organs
#SBATCH --output=/home/achechenina/results/slurm_out/cana_organs_%j.out
#SBATCH --error=/home/achechenina/results/slurm_out/cana_organs_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=normal

set -euo pipefail
mkdir -p /home/achechenina/results/slurm_out
echo "Starting canagliflozin organ calls at $(date) on $(hostname)"
source /net/fs001/pool0/home/achechenina/miniforge3/etc/profile.d/conda.sh
set +u
conda activate ds_python310_env
set -u
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python -u /data/labs/sinha_lab/achechenina/projects/organ_segm_class_model/src/score_jax_itp_canagliflozin.py
echo "Finished at $(date)"
