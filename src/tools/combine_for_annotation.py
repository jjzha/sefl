import pandas as pd
import random

random.seed(42)

# Example file lists (adjust paths/names to your actual files)
model_a_files = [
    "src/analysis/data/Qwen2.5-0.5B-Instruct-SEFI_seed1_inference.csv", 
    "src/analysis/data/Llama-3.2-1B-Instruct-SEFI_seed2_inference.csv", 
    "src/analysis/data/Llama-3.2-3B-Instruct-SEFI_seed3_inference.csv", 
    "src/analysis/data/Llama-3.1-8B-Instruct-SEFI_seed4_inference.csv", 
    "src/analysis/data/Qwen2.5-14B-Instruct-SEFI_seed5_inference.csv"
]
model_b_files = [
    "src/analysis/data/Qwen-Qwen2.5-0.5B-Instruct_seed1_inference.csv", 
    "src/analysis/data/meta-llama-Llama-3.2-1B-Instruct_seed2_inference.csv", 
    "src/analysis/data/meta-llama-Llama-3.2-3B-Instruct_seed3_inference.csv", 
    "src/analysis/data/meta-llama-Llama-3.1-8B-Instruct_seed4_inference.csv", 
    "src/analysis/data/Qwen-Qwen2.5-14B-Instruct_seed5_inference.csv"
]

combined_rows = []

for a_file, b_file in zip(model_a_files, model_b_files):
    df_a = pd.read_csv(a_file)
    df_b = pd.read_csv(b_file)
    
    if len(df_a) != len(df_b):
        raise ValueError(f"Files {a_file} and {b_file} do not have the same number of rows.")
    
    for idx in range(len(df_a)):
        appendix_a = df_a.at[idx, "appendix"]
        prompt_a = df_a.at[idx, "prompt"]
        feedback_a = df_a.at[idx, "feedback"]
        
        appendix_b = df_b.at[idx, "appendix"]
        prompt_b = df_b.at[idx, "prompt"]
        feedback_b = df_b.at[idx, "feedback"]
        
        if prompt_a != prompt_b:
            raise ValueError(
                f"Mismatch in prompts at row {idx}:\n"
                f"{a_file} prompt: {prompt_a}\n"
                f"{b_file} prompt: {prompt_b}"
            )

        # Randomly decide whether to keep A in the "model_a" slot 
        # or to swap with B
        if random.choice([True, False]):
            row = {
                "appendix": appendix_a,
                "prompt":  prompt_a,
                "model_a": feedback_a,
                "model_b": feedback_b,
                "source_a": a_file,
                "source_b": b_file,
            }
        else:
            row = {
                "appendix": appendix_a,
                "prompt":  prompt_a,
                "model_a": feedback_b,
                "model_b": feedback_a,
                "source_a": b_file,
                "source_b": a_file,
            }
        combined_rows.append(row)

combined_df = pd.DataFrame(combined_rows)

combined_df.to_csv("annotation_file.csv", index=False)
print("Combined annotation file created: annotation_file.csv")