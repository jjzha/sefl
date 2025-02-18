import sys
import os
import json
from huggingface_hub import HfApi, login

def main():
    # Hard-code or retrieve your HF token however you prefer
    hf_token = ""
    
    # Log into Hugging Face Hub once per process
    login(token=hf_token)
    api = HfApi()

    # The directory name is passed in as the first argument
    if len(sys.argv) < 2:
        print("Usage: python push_to_hub.py <directory_name>")
        sys.exit(1)

    dir_name = sys.argv[1]
    local_folder = f"final_2/{dir_name}"

    # 2) Upload the entire folder
    api.upload_large_folder(
        folder_path=local_folder,
        repo_id=f"xxxxx/{dir_name}",
        repo_type="model",
        private=True,
    )

    print(f"[INFO] Finished uploading {dir_name} to HF.")

if __name__ == "__main__":
    main()
