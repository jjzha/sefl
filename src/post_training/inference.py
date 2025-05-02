import argparse
import csv
import torch
from datasets import load_dataset
from transformers import pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Name of the chat-friendly model on Hugging Face."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Name of your dataset on Hugging Face."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        help="Which split of the dataset to load."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of random samples from the dataset."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed for random samples."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="inference_results.csv",
        help="Where to save the CSV of (prompt, feedback)."
    )
    args = parser.parse_args()

    system_prompt = (
        """You are a skilled teacher specializing in creating concise, effective, and targeted feedback. Your key responsibilities are:\n
        Feedback Provision:\n
        1. Offer constructive feedback on completed work.\n
        2. explain concepts succinctly when needed, do not give grades, only feedback for each mistake.\n
        3. Encouragement and Adaptation: Encourage critical thinking and creativity; adapt to different learning styles and levels.\n
        Your goal is to facilitate learning through well-designed tasks and helpful guidance.
        You receive the student assignment and the answers to the assignment, give feedback in 200 words or less."""
    )

    device_index = 0 if torch.cuda.is_available() else -1
    print(f"Using device index: {device_index}")
    if device_index >= 0:
        print(f"Device name is {torch.cuda.get_device_name(device_index)}")

    dataset = load_dataset(args.dataset, split=args.split)
    dataset = dataset.shuffle(seed=args.seed).select(range(args.num_samples))

    pipe = pipeline(
        "text-generation",
        model=args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        continue_final_message=True,
    )

    with open(args.output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["appendix", "prompt", "feedback"])  # CSV header

        for i, sample in enumerate(dataset):
            appendix = "### ASSIGNMENT APPENDIX\n" + sample["pretrain"]
            user_text = sample["prompt"]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": appendix + "\n\n" + user_text},
                {"role": "assistant", "content": "### FEEDBACK: "}
            ]

            results = pipe(
                messages,
                max_new_tokens=1024
            )
            feedback = results[0]["generated_text"][-1]["content"]

            writer.writerow([appendix, user_text, feedback])

            print(f"=== Sample {i+1} ===")
            print(f"PROMPT:\n{user_text}")
            print(f"FEEDBACK:\n{feedback}")
            print("---------------------")

    print(f"\nDone! Results saved in {args.output_csv}")

if __name__ == "__main__":
    main()
