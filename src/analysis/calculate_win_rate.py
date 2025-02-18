import pandas as pd
import argparse


def calculate_winrate(verdict_file, annotation_file):
    """
    Calculate the win rate percentage for models based on verdicts and source mappings.
    
    Parameters:
        verdict_file (str): Path to the CSV file with the verdicts.
        annotation_file (str): Path to the CSV file with the source mappings.
    
    Returns:
        pd.DataFrame: Win rate percentages for each model.
    """
    # Load the verdict and annotation files
    verdict_df = pd.read_csv(verdict_file)
    annotation_df = pd.read_csv(annotation_file)
    
    # Merge the two dataframes on the prompt (ensure the prompts align)
    merged_df = pd.merge(
        verdict_df,
        annotation_df,
        on="prompt",
        suffixes=("_verdict", "_source"),
    )

    # Initialize counters
    win_counts = {}
    total_counts = {}

    # Process each row in the merged DataFrame
    for _, row in merged_df.iterrows():
        source_a = row["source_a"]
        source_b = row["source_b"]
        verdict = row["verdict"]

        # Update total counts for both models
        total_counts[source_a] = total_counts.get(source_a, 0) + 1
        total_counts[source_b] = total_counts.get(source_b, 0) + 1

        # Update win counts based on the verdict
        if verdict == "A":
            win_counts[source_a] = win_counts.get(source_a, 0) + 1
        elif verdict == "B":
            win_counts[source_b] = win_counts.get(source_b, 0) + 1

    # Calculate win rates
    win_rate_data = []
    for model, total in total_counts.items():
        wins = win_counts.get(model, 0)
        win_rate = (wins / total) * 100 if total > 0 else 0
        win_rate_data.append({"model": model, "wins": wins, "total": total, "win_rate": win_rate})

    # Convert to DataFrame and sort by win rate
    win_rate_df = pd.DataFrame(win_rate_data).sort_values(by="win_rate", ascending=False)
    return win_rate_df


def main():
    # Argument parser for dynamic input
    parser = argparse.ArgumentParser(description="Calculate win rates for models based on verdicts.")
    parser.add_argument("--verdict_file", type=str, required=True, help="Path to the verdict CSV file.")
    parser.add_argument("--annotation_file", type=str, required=True, help="Path to the annotation CSV file.")
    parser.add_argument("--output_file", type=str, default="win_rates.csv", help="Output file for win rates.")
    args = parser.parse_args()

    # Calculate win rates
    win_rate_df = calculate_winrate(args.verdict_file, args.annotation_file)

    # Save to CSV
    win_rate_df.to_csv(args.output_file, index=False)
    print(f"Win rates saved to: {args.output_file}")


if __name__ == "__main__":
    main()
