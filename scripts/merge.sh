#!/bin/bash
#SBATCH --partition=standard-g
#SBATCH --account=project_xxxxxxxxxx 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem-per-gpu=60G 
#SBATCH --cpus-per-task=56
#SBATCH --time=02:00:00
#SBATCH --job-name=merge_all
#SBATCH --output=/scratch/project_xxxxxxxxxx/synthetic-edu-cache/merge_all_%A.out

# Debug mode for verbose output
set -x

# Load modules
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

# Container
CONTAINER=/scratch/project_xxxxxxxxxx/synthetic-edu/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif

# Arrays of checkpoint dirs, output paths, and model dirs
CHECKPOINT_DIRS=(
    "/scratch/project_xxxxxxxxxx/synthetic-edu-cache/tmp/Llama-3.2-3B-Instruct-sft-synthetic/checkpoint-36/pytorch_model_fsdp_0/"
    "/scratch/project_xxxxxxxxxx/synthetic-edu-cache/tmp/Llama-3.1-8B-Instruct-sft-synthetic/checkpoint-36/pytorch_model_fsdp_0/"
    "/scratch/project_xxxxxxxxxx/synthetic-edu-cache/tmp/Qwen2.5-14B-Instruct-sft-synthetic/checkpoint-36/pytorch_model_fsdp_0/"
)
OUTPUT_DIRS=(
    "/scratch/project_xxxxxxxxxx/synthetic-edu-cache/upload/Llama-3.2-3B-Instruct-SEFI/"
    "/scratch/project_xxxxxxxxxx/synthetic-edu-cache/upload/Llama-3.1-8B-Instruct-SEFI/"
    "/scratch/project_xxxxxxxxxx/synthetic-edu-cache/upload/Qwen2.5-14B-Instruct-SEFI/"
)

MAX_SHARD_SIZE="10GB"

# Loop over the array indices
for i in "${!CHECKPOINT_DIRS[@]}"; do
    srun singularity exec -B /scratch/project_xxxxxxxxxx/synthetic-edu-cache $CONTAINER \
         bash -c "\
           \$WITH_CONDA; \
           python3 tools/merge.py \
             --checkpoint_dir '${CHECKPOINT_DIRS[i]}' \
             --output_path '${OUTPUT_DIRS[i]}' \
             --max_shard_size '${MAX_SHARD_SIZE}' \
         " &
done

# Wait for all background jobs to finish
wait

echo "All merges are complete!"
