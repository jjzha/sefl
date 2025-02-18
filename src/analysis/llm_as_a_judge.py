#!/usr/bin/env python3

"""
Example usage:
   python custom_model_as_a_judge.py --input_csv annotation_file.csv \
                        --output_csv results_custom.csv
"""

import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Use a custom Hugging Face model to judge which feedback is better: A or B.")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to the CSV with columns: prompt, model_a, model_b, etc.")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to save the output CSV with verdicts.")
    parser.add_argument("--hf_model", type=str, required=True,
                        help="Hugging Face model name (e.g., 'Skywork/Skywork-Critic-Llama3.1-70B').")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load the custom model and tokenizer
    model_name = args.hf_model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Read the input CSV
    df = pd.read_csv(args.input_csv)
    results = []

    for idx, row in df.iterrows():
        appendix = row.get("pretrain", "")
        prompt = row.get("prompt", "")
        model_a_feedback = row.get("model_a", "")
        model_b_feedback = row.get("model_b", "")

        user_message = (
            "You are tasked with evaluating assignment feedback provided by two different models (Model A and Model B). As an objective evaluator, follow these steps: 1. Analysis Criteria: - Accuracy: Does the feedback directly address specific strengths and weaknesses without unnecessary elaboration? - Actionability: Are suggestions clear, specific, and implementable without being overly prescriptive? - Conciseness: Is the feedback brief and focused while remaining meaningful? - Tone: Does the feedback maintain efficiency while being constructive? 2. Evaluation Process: - First, review the original assignment task carefully - Then examine both Model A's and Model B's feedback responses - Compare them against the above criteria - Prioritize focused, efficient feedback over exhaustive detail 3. Scoring Rules: - Responses should not include numerical grades - Feedback must be concise and directly related to the student's work - Each point should be essential and identify specific aspects of the response - Avoid unnecessary categorization and theoretical benefits 4. Output Format: - Respond with a single character: 'A' or 'B' - Choose the model that provides more targeted, efficient feedback - Do not provide any additional explanation or commentary - Your response must contain exactly one character.\n\n"
            f"[User Question]\n### ASSIGNMENT APPENDIX\n{appendix}\n\n"
            f"\n{prompt}\n\n"
            f"[The Start of Assistant A's Answer]\n{model_a_feedback}[The End of Assistant A's Answer]\n\n"
            f"[The Start of Assistant B's Answer]\n{model_b_feedback}[The End of Assistant B's Answer]\n\n"
            "Which is better? Please respond with a single character: A or B."
        )
        # Create the conversation format
        conversation = [{"role": "user", "content": user_message}]

        try:
            # Tokenize and prepare input
            input_ids = tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            # Generate the output
            generation = model.generate(
                input_ids=input_ids,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                temperature=0
            )
        
            # Decode the response
            raw_text = tokenizer.decode(
                generation[0][len(input_ids[0]):],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True).strip()

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            raw_text = "ERROR"

        # Extract '[[A]]' or '[[B]]' from the raw_text (fall back to 'UNKNOWN' if not found)
        verdict_char = "UNKNOWN"
        if "A" in raw_text:
            verdict_char = "A"
        elif "B" in raw_text:
            verdict_char = "B"

        results.append({
            "idx": idx,
            "prompt": prompt,
            "model_a": model_a_feedback,
            "model_b": model_b_feedback,
            "verdict": verdict_char,
            "raw_response": raw_text
        })

    # Save the results
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Done. Wrote results to: {args.output_csv}")

if __name__ == "__main__":
    main()
