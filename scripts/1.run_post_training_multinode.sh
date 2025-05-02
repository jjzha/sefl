#!/bin/bash -l
#SBATCH --job-name=syntethic_post_training
#SBATCH --output=/scratch/project_xxxxxxxxxx/synthetic-edu-cache/synthetic_%A_%a.out
#SBATCH --error=/scratch/project_xxxxxxxxxx/synthetic-edu-cache/synthetic_%A_%a.ERROR.out
#SBATCH --partition=standard-g 
#SBATCH --account=project_xxxxxxxxxx
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclude=nid005003,nid007971,nid007972


module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits
CONTAINER=/scratch/project_xxxxxxxxxx/synthetic-edu/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif

SCRATCH=/scratch/project_xxxxxxxxxx/synthetic-edu-cache
export TORCH_HOME=$SCRATCH/.torch-cache
export HF_HOME=$SCRATCH/.hf-cache-multinode
mkdir -p $TORCH_HOME $HF_HOME
export TOKENIZERS_PARALLELISM=false

# Set network interfaces for RCCL/NCCL
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB

# Define the models you want to train
MODELS=()

NUM_PROCESSES=$(expr $SLURM_NNODES \* $SLURM_GPUS_PER_NODE)
MAIN_PROCESS_IP=$(hostname -i)

for MODEL in "${MODELS[@]}"
do
    echo "Training model: $MODEL"
    echo "Removing old cache..."
    rm -rf $HF_HOME/datasets

    LAUNCH_CMD="
    accelerate launch \
        --config_file=configs/accelerate_hf_trainer_config_fsdp.yaml \
        --num_machines=$SLURM_NNODES \
        --num_processes=$NUM_PROCESSES \
        --machine_rank=\$SLURM_NODEID \
        --main_process_ip=$MAIN_PROCESS_IP \
      src/post_training/fine_tune.py \
        --model_name_or_path $MODEL \
        --dataset_name ""
        --packing \
        --bf16 \
        --learning_rate 2.0e-5 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --gradient_checkpointing \
        --logging_strategy steps \
        --logging_steps 1 \
        --eval_strategy epoch \
        --save_strategy epoch \
        --output_dir $SCRATCH/model_checkpoints/$(basename $MODEL | tr / -)-sft-synthetic
    "

    srun singularity exec -B /scratch/project_xxxxxxxxxx/ \
        $CONTAINER \
         bash -c "\$WITH_CONDA; \
                export TORCH_HOME=$TORCH_HOME; \
                export HF_HOME=$HF_HOME; \
                $LAUNCH_CMD"

done
