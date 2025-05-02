#!/bin/sh
#SBATCH --partition=standard-g
#SBATCH --account=project_xxxxxxxxxx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-gpu=60G
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=push_model_to_hub
#SBATCH --output=/scratch/project_xxxxxxxxxx/synthetic-edu-cache/push_to_hub_%A_%a.out

set -x

module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/scratch/project_xxxxxxxxxx/synthetic-edu/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif

DIRS=(
    "Llama-3.1-8B-Instruct-SEFI"
    "Llama-3.2-3B-Instruct-SEFI"
    "Qwen2.5-0.5B-Instruct-SEFI"
    "Qwen2.5-1.5B-Instruct-SEFI"
    "Qwen2.5-14B-Instruct-SEFI"
)

for dir_name in "${DIRS[@]}"; do
    echo "[INFO] Now processing: ${dir_name}"
    srun --ntasks=1 --exclusive singularity exec -B /scratch/project_xxxxxxxxxx/synthetic-edu-cache "$CONTAINER" \
        bash -c "\$WITH_CONDA; python3 tools/push_to_hub.py ${dir_name}"
done
