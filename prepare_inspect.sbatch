#!/usr/bin/env bash
#SBATCH --job-name=gpt2-prepare-inspect
#SBATCH --account=acd115198
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/prepare-inspect-%j.out
#SBATCH --error=logs/prepare-inspect-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
echo "Allocated job=$SLURM_JOB_ID node=$SLURMD_NODENAME"
srun --ntasks=1 --kill-on-bad-exit=1 bash -c '
    set -euo pipefail
    uv sync --locked --link-mode=copy
    if [[ ! -f data/manifest.json ]]; then
        uv run --no-sync python -u prepare_data.py
    fi
    uv run --no-sync python -u inspect_attention.py
'
