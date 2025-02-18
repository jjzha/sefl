#!/bin/bash
#SBATCH --job-name=llm_judge
#SBATCH --output=/scratch/project_xxxxxxxxxx/synthetic-edu-cache/llm_judge_%j.out
#SBATCH --error=/scratch/project_xxxxxxxxxx/synthetic-edu-cache/llm_judge_%j.err
#SBATCH --partition=standard-g
#SBATCH --account=project_xxxxxxxxxx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=60G
#SBATCH --cpus-per-task=56
#SBATCH --time=04:00:00

# ------------------------------
# Setup environment
# ------------------------------
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/scratch/project_xxxxxxxxxx/synthetic-edu/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif

SCRATCH=/scratch/project_xxxxxxxxxx/synthetic-edu-cache
export TORCH_HOME=$SCRATCH/.torch-cache
export HF_HOME=$SCRATCH/.hf-cache-reward
export TOKENIZERS_PARALLELISM=false

mkdir -p $TORCH_HOME $HF_HOME
INPUT_CSV="/scratch/project_xxxxxxxxxx/synthetic-edu/src/analysis/data/annotation_file.csv"
OUTPUT_CSV="/scratch/project_xxxxxxxxxx/synthetic-edu-cache/tmp_2/results_internlm.csv"

echo "Starting Llama judge script..."
# The 'srun' command will launch the container with GPU support.
rm -rf $HF_HOME

srun singularity exec -B /scratch/project_xxxxxxxxxx/ \
        $CONTAINER \
        bash -c "\$WITH_CONDA; \
                export TORCH_HOME=$TORCH_HOME; \
                export HF_HOME=$HF_HOME; \
                python src/analysis/llm_as_a_judge.py \
                        --input_csv $INPUT_CSV \
                        --output_csv $OUTPUT_CSV \
                        --hf_model internlm/internlm2-20b-reward \
                        "
echo "Done."