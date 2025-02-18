#!/bin/sh
#SBATCH --partition=standard-g
#SBATCH --account=project_xxxxxxxxxx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=60G
#SBATCH --cpus-per-task=56
#SBATCH --time=3:00:00
#SBATCH --job-name=synthetic_inference
#SBATCH --output=/scratch/project_xxxxxxxxxx/synthetic-edu-cache/synthetic_inference_%A_%a.out
#SBATCH --error=/scratch/project_xxxxxxxxxx/synthetic-edu-cache/synthetic_inference_%A_%a_ERROR.out

# ---------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/scratch/project_xxxxxxxxxx/synthetic-edu/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif

SCRATCH=/scratch/project_xxxxxxxxxx/synthetic-edu-cache
export TORCH_HOME=$SCRATCH/.torch-cache
export HF_HOME=$SCRATCH/.hf-cache-inference
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TORCH_HOME" "$HF_HOME"

# ---------------------------------------------------------------------
# Define paired vanilla and tuned models (same indices)
# ---------------------------------------------------------------------
VANILLA_MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
)

TUNED_MODELS=(
    "jjzha/Qwen2.5-0.5B-Instruct-SEFI"
    "jjzha/Llama-3.2-1B-Instruct-SEFI"
    "jjzha/Llama-3.2-3B-Instruct-SEFI"
    "jjzha/Llama-3.1-8B-Instruct-SEFI"
    "jjzha/Qwen2.5-14B-Instruct-SEFI"
)

# ---------------------------------------------------------------------
# Loop over each pair, incrementing the seed from 1 up to N
# ---------------------------------------------------------------------
for i in "${!VANILLA_MODELS[@]}"; do
    
    # The vanilla and tuned models at index i
    VANILLA_MODEL="${VANILLA_MODELS[$i]}"
    TUNED_MODEL="${TUNED_MODELS[$i]}"

    # The seed we want to use for this pair is i+1
    SEED=$((i+1))
    echo "Removing old cache..."    
    rm -rf $HF_HOME/datasets
    # ---------------------
    # 1. Inference: Vanilla
    # ---------------------
    echo "=== Pair #$((i+1)): $VANILLA_MODEL with seed=$SEED ==="

    # Generate a filename for the vanilla model output
    CSV_NAME_VANILLA="$(echo "$VANILLA_MODEL" | tr '/' '-')_seed${SEED}_inference.csv"
    OUTPUT_PATH_VANILLA="$SCRATCH/tmp_2/$CSV_NAME_VANILLA"

    srun singularity exec -B /scratch/project_xxxxxxxxxx/ \
        "$CONTAINER" \
        bash -c "\$WITH_CONDA; \
            export TORCH_HOME=$TORCH_HOME; \
            export HF_HOME=$HF_HOME; \
            python3 src/post_training/inference.py \
                --model \"$VANILLA_MODEL\" \
                --dataset jjzha/synthetic-feedback \
                --split valid \
                --num_samples 30 \
                --seed $SEED \
                --output_csv \"$OUTPUT_PATH_VANILLA\" \
            "

    echo "=== Finished inference for $VANILLA_MODEL ==="
    echo "Results are in: $OUTPUT_PATH_VANILLA"
    echo

    # ---------------------
    # 2. Inference: Tuned
    # ---------------------
    echo "=== Pair #$((i+1)): $TUNED_MODEL with seed=$SEED ==="

    CSV_NAME_TUNED="$(echo "$TUNED_MODEL" | tr '/' '-')_seed${SEED}_inference.csv"
    OUTPUT_PATH_TUNED="$SCRATCH/tmp_2/$CSV_NAME_TUNED"

    srun singularity exec -B /scratch/project_xxxxxxxxxx/ \
        "$CONTAINER" \
        bash -c "\$WITH_CONDA; \
            export TORCH_HOME=$TORCH_HOME; \
            export HF_HOME=$HF_HOME; \
            python3 src/post_training/inference.py \
                --model \"$TUNED_MODEL\" \
                --dataset jjzha/synthetic-feedback \
                --split valid \
                --num_samples 30 \
                --seed $SEED \
                --output_csv \"$OUTPUT_PATH_TUNED\" \
            "

    echo "=== Finished inference for $TUNED_MODEL ==="
    echo "Results are in: $OUTPUT_PATH_TUNED"
    echo

done

echo "All inferences have completed!"
