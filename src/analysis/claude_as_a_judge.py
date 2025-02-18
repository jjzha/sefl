#!/usr/bin/env python3

"""
Example usage:
   python claude_as_a_judge.py --input_csv annotation_file.csv \
                        --output_csv results_gpt4.csv
   (Ensure ANTHROPIC_API_KEY is set in the environment.)
"""

import os
import anthropic
import argparse
import pandas as pd
from time import sleep

def parse_args():
    parser = argparse.ArgumentParser(description="Use GPT-4 to judge which feedback is better: A or B.")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to the CSV with columns: prompt, model_a, model_b, etc.")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to save the output CSV with verdicts.")
    parser.add_argument("--gpt_model", type=str, default="claude-3-5-sonnet-20241022",
                        help="model name.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    # Read the input CSV
    df = pd.read_csv(args.input_csv)
    results = []

    for idx, row in df.iterrows():
        appendix = row.get("pretrain", "")
        prompt = row.get("prompt", "")
        model_a_feedback = row.get("model_a", "")
        model_b_feedback = row.get("model_b", "")

        # Construct a system+user prompt instructing GPT-4 to pick either A or B
        messages = [
            {
                "role": "user",
                "content": (
                    "You are tasked with evaluating assignment feedback provided by two different models (Model A and Model B). As an objective evaluator, follow these steps: 1. Analysis Criteria: - Accuracy: Does the feedback directly address specific strengths and weaknesses without unnecessary elaboration? - Actionability: Are suggestions clear, specific, and implementable without being overly prescriptive? - Conciseness: Is the feedback brief and focused while remaining meaningful? - Tone: Does the feedback maintain efficiency while being constructive? 2. Evaluation Process: - First, review the original assignment task carefully - Then examine both Model A's and Model B's feedback responses - Compare them against the above criteria - Prioritize focused, efficient feedback over exhaustive detail 3. Scoring Rules: - Responses should not include numerical grades - Feedback must be concise and directly related to the student's work - Each point should be essential and identify specific aspects of the response - Avoid unnecessary categorization and theoretical benefits 4. Output Format: - Respond with a single character: 'A' or 'B' - Choose the model that provides more targeted, efficient feedback - Do not provide any additional explanation or commentary - Your response must contain exactly one character.\n\n"
                    f"Assignment Appendix:\n### ASSIGNMENT APPENDIX\n{appendix}\n\n"
                    f"Assignment Prompt:\n{prompt}\n\n"
                    f"Model A feedback:\n{model_a_feedback}\n\n"
                    f"Model B feedback:\n{model_b_feedback}\n\n"
                    "Which is better? Please respond with a single character: A or B."
                )
            }
        ]

        try:
            response = client.messages.create(
                model=args.gpt_model,
                messages=messages,
                max_tokens=2      # We only need a single character
            )
            raw_text = response.content[-1].text.strip()

        except Exception as e:
            print(f"Error calling Anthropic API on row {idx}: {e}")
            raw_text = "ERROR"

        # Extract 'A' or 'B' from the raw_text (fall back to 'UNKNOWN' if not found)
        verdict_char = "UNKNOWN"
        if raw_text.upper().startswith("A"):
            verdict_char = "A"
        elif raw_text.upper().startswith("B"):
            verdict_char = "B"

        results.append({
            "idx": idx,
            "prompt": prompt,
            "model_a": model_a_feedback,
            "model_b": model_b_feedback,
            "verdict": verdict_char,
            "raw_response": raw_text
        })

        # Optional: delay to avoid hitting rate limits
        sleep(0.5)

    # Save the results
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Done. Wrote results to: {args.output_csv}")

if __name__ == "__main__":
    main()
