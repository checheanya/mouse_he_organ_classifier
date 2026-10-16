#!/bin/bash
# Train the reusable organ classifier on unrefined annotated segmentations.
# Spleen and pancreas are not included. CPU only. A GPU does not help:
# the work is mean-pooling UNI vectors and fitting a logistic regression.
#
#SBATCH --job-name=organ-cls
#SBATCH --output=/home/achechenina/results/slurm_out/organ_cls_%j.out
#SBATCH --error=/home/achechenina/results/slurm_out/organ_cls_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=normal

set -euo pipefail
echo "Starting organ classifier at $(date) on $(hostname)"
source /net/fs001/pool0/home/achechenina/miniforge3/etc/profile.d/conda.sh
set +u
conda activate ds_python310_env
set -u
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ROOT=/data/labs/sinha_lab/achechenina/projects/organ_segm_class_model
cd "$ROOT"
mkdir -p /home/achechenina/results/slurm_out "$ROOT/results"
python -u "$ROOT/src/build_annotations.py"
python -u "$ROOT/src/train.py" --workers "${SLURM_CPUS_PER_TASK}"
echo "Finished at $(date)"
