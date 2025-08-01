#!/bin/bash -l
#SBATCH --job-name=gpt4_judge
#SBATCH --output=/scratch/project_xxxxxxxxxx/synthetic-edu-cache/gpt4_judge_%j.out
#SBATCH --error=/scratch/project_xxxxxxxxxx/synthetic-edu-cache/gpt4_judge_%j.err
#SBATCH --partition=standard
#SBATCH --account=project_xxxxxxxxxx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

CONTAINER=/scratch/project_xxxxxxxxxx/synthetic-edu/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

export OPENAI_API_KEY="XXXX"

SCRATCH=/scratch/project_xxxxxxxxxx/synthetic-edu-cache
export TORCH_HOME=$SCRATCH/.torch-cache
export HF_HOME=$SCRATCH/.hf-cache-gpt4-judge
mkdir -p $TORCH_HOME $HF_HOME

INPUT_CSV=""
OUTPUT_CSV=""

echo "Starting GPT-4 judge script..."

srun singularity exec -B /scratch/project_xxxxxxxxxx/ \
     "$CONTAINER" \
     bash -c "\
        \$WITH_CONDA; \
        export TORCH_HOME=$TORCH_HOME; \
        export HF_HOME=$HF_HOME; \
        python src/analysis/gpt4o_as_a_judge.py \
            --input_csv $INPUT_CSV \
            --output_csv $OUTPUT_CSV \
            --gpt_model gpt-4o \
     "

echo "Done."
