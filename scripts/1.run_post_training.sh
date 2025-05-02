#!/bin/sh
#SBATCH --partition=standard-g 
#SBATCH --account=project_xxxxxxxxx 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem-per-gpu=60G 
#SBATCH --cpus-per-task=56
#SBATCH --time=2:00:00
#SBATCH --job-name=syntethic_post_training
#SBATCH --output=/scratch/project_xxxxxxxxx/synthetic-edu-cache/synthetic_%A_%a.out
#SBATCH --error=/scratch/project_xxxxxxxxx/synthetic-edu-cache/synthetic_%A_%a.ERROR.out


module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/scratch/project_xxxxxxxxx/synthetic-edu/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
SCRATCH=/scratch/project_xxxxxxxxx/synthetic-edu-cache

export TORCH_HOME=$SCRATCH/.torch-cache
export HF_HOME=$SCRATCH/.hf-cache
export TOKENIZERS_PARALLELISM=false
mkdir -p $TORCH_HOME $HF_HOME

MODELS=()

for MODEL in "${MODELS[@]}"
do
    echo "Training model: $MODEL"
    echo "Removing old cache..."
    rm -rf $HF_HOME/datasets

    srun singularity exec -B /scratch/project_xxxxxxxxx/ \
         $CONTAINER \
            bash -c "\$WITH_CONDA; \
                     export TORCH_HOME=$TORCH_HOME; \
                     export HF_HOME=$HF_HOME; \
                     accelerate launch \
                        --config_file=configs/accelerate_hf_trainer_config.yaml \
                        --num_machines=1 \
                        --num_processes=${SLURM_GPUS_PER_NODE} \
                        --machine_rank=0 \
                        src/post_training/fine_tune.py \
                            --model_name_or_path $MODEL \
                            --dataset_name xxxxxxx \ ### CHANGE DATASET
                            --learning_rate 2.0e-5 \
                            --num_train_epochs 3 \
                            --packing \
                            --bf16 \
                            --per_device_train_batch_size 4 \
                            --gradient_accumulation_steps 4 \
                            --gradient_checkpointing \
                            --logging_strategy=\"steps\" \
                            --logging_steps 1 \
                            --eval_strategy epoch \
                            --save_strategy epoch \
                            --output_dir $SCRATCH/model_checkpoints/$(basename $MODEL | tr / -)-sft-synthetic" \
                            
done
