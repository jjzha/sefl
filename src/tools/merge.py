#!/usr/bin/env python3

import argparse
from accelerate.utils import merge_fsdp_weights
from transformers import AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="Merge FSDP weights and save model.")

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the FSDP sharded checkpoint directory."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where merged weights (checkpoint) will be stored."
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="10GB",
        help="Shard size for saving the model (e.g., 10GB, 5GB, etc.)."
    )

    args = parser.parse_args()

    # Merge sharded FSDP weights into a single checkpoint
    merge_fsdp_weights(
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output_path
    )

    # Load the model from the merged weights
    model = AutoModelForCausalLM.from_pretrained(
        args.output_path,
        device_map="auto"  # Optional: automatically uses available GPUs
    )

    # Save the fully merged model
    model.save_pretrained(
        args.output_path,
        safe_serialization=True,      # to keep using safetensors, if desired
        max_shard_size=args.max_shard_size
    )

if __name__ == "__main__":
    main()
