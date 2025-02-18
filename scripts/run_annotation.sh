#!/bin/bash -l
#SBATCH --job-name=win-rate
#SBATCH --output=/scratch/project_xxxxxxxxxx/synthetic-edu-cache/annotation_%j.out
#SBATCH --error=/scratch/project_xxxxxxxxxx/synthetic-edu-cache/annotation_%j.err
#SBATCH --partition=standard
#SBATCH --account=project_xxxxxxxxxx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

# ------------------------------
# Singularity container path
# ------------------------------
CONTAINER=/scratch/project_xxxxxxxxxx/synthetic-edu/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif

# ------------------------------
# HPC Modules (if needed)
# ------------------------------
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

# ------------------------------
# Run command inside container
# ------------------------------
echo "Starting Annotation generation script..."

srun singularity exec -B /scratch/project_xxxxxxxxxx/ \
     "$CONTAINER" \
     bash -c "\
        \$WITH_CONDA; \
        python src/tools/combine_for_annotation.py
    "

echo "Done."
