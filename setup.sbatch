#!/usr/bin/env bash
#SBATCH --job-name=gpt2-uv-setup
#SBATCH --account=acd115198
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/setup-%j.out
#SBATCH --error=logs/setup-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "Allocated job=$SLURM_JOB_ID node=$SLURMD_NODENAME"
srun --ntasks=1 --kill-on-bad-exit=1 bash -c '
    set -euo pipefail
    uv lock
    uv sync --locked --link-mode=copy
'
