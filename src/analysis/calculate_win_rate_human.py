import pandas as pd

def compute_model_human_stats(keys_csv, human_csv_files, output_csv):
    """
    Computes statistics from the model perspective.
    
    For each candidate model (from keys.csv columns A and B), for each human annotator,
    this function computes:
      - win_rate: Fraction of assignments (where the model appeared) in which the annotator voted for that model.
      - sense_rate: Average (binary) sense rating for assignments where the model appeared (with "yes" = 1, else 0).
      - comments: Consolidated annotator comments (concatenated string) for that candidate.
      
    The output CSV will have one row per candidate model and columns for each human annotator such as:
      human1_win_rate, human1_sense_rate, human1_comments, human2_win_rate, human2_sense_rate, human2_comments, etc.
    """
    # === Step 1. Load keys.csv ===
    df_keys = pd.read_csv(keys_csv)
    df_keys['Assignment'] = df_keys['Assignment'].astype(str).str.strip()
    # Expect keys.csv to have columns: Assignment, A, B

    # === Step 2. Load and tag human annotation files ===
    human_dfs = []
    for i, file in enumerate(human_csv_files, start=1):
        df = pd.read_csv(file)
        # Clean the key columns so that merging works.
        df['Assignment'] = df['Assignment'].astype(str).str.strip()
        df['Preferred'] = df['Preferred'].astype(str).str.strip().str.upper()
        df['Sense'] = df['Sense'].astype(str).str.strip().str.lower()
        # Comments: leave as string (if missing, will be NaN)
        # Add a column to identify the annotator.
        df['human'] = f"human{i}"
        human_dfs.append(df)
    df_humans = pd.concat(human_dfs, ignore_index=True)
    
    # === Step 3. Merge human annotations with keys on Assignment ===
    df_merged = pd.merge(df_humans, df_keys, on='Assignment', how='inner')
    # Now each row has:
    #   - Assignment, Preferred, Sense, Comments, human (from the human file)
    #   - A and B (from keys.csv, the candidate models for that assignment)
    
    # === Step 4. Transform to candidate-level records ===
    # For each merged row, create two records:
    #   - One for the candidate from column A, with win = 1 if Preferred == "A", else 0.
    #   - One for the candidate from column B, with win = 1 if Preferred == "B", else 0.
    # Also, convert Sense to binary (1 if "yes", else 0) and carry along the Comments.
    def convert_sense(s):
        return 1 if s == "yes" else 0

    df_merged['sense_binary'] = df_merged['Sense'].apply(convert_sense)
    
    # Candidate from column A.
    df_A = df_merged[['Assignment', 'human', 'Preferred', 'sense_binary', 'Comments', 'A']].copy()
    df_A = df_A.rename(columns={'A': 'candidate'})
    df_A['win'] = (df_A['Preferred'] == 'A').astype(int)
    
    # Candidate from column B.
    df_B = df_merged[['Assignment', 'human', 'Preferred', 'sense_binary', 'Comments', 'B']].copy()
    df_B = df_B.rename(columns={'B': 'candidate'})
    df_B['win'] = (df_B['Preferred'] == 'B').astype(int)
    
    # Combine the candidate-level records.
    df_long = pd.concat([df_A, df_B], ignore_index=True)
    
    # === Step 5. Group by candidate model and human annotator ===
    # For each candidate/human group, we calculate:
    #   - appearances: count of assignments in which the candidate was an option.
    #   - wins: sum of win flags.
    #   - total_sense: sum of sense_binary (i.e. count of assignments with "yes").
    #   - comments: consolidate all comments from that human for that candidate.
    stats = df_long.groupby(['candidate', 'human']).agg(
        appearances=('Assignment', 'count'),
        wins=('win', 'sum'),
        total_sense=('sense_binary', 'sum'),
        comments=('Comments', lambda x: " | ".join(x.dropna().astype(str).unique()))
    ).reset_index()
    
    # Compute win_rate and sense_rate.
    stats['win_rate'] = stats['wins'] / stats['appearances']
    stats['sense_rate'] = stats['total_sense'] / stats['appearances']
    
    # === Step 6. Pivot the results so that rows are candidate models and columns are per-human statistics.
    # Pivot for win_rate.
    win_rate_pivot = stats.pivot(index='candidate', columns='human', values='win_rate')
    win_rate_pivot = win_rate_pivot.rename(columns=lambda x: f"{x}_win_rate")
    # Pivot for sense_rate.
    sense_rate_pivot = stats.pivot(index='candidate', columns='human', values='sense_rate')
    sense_rate_pivot = sense_rate_pivot.rename(columns=lambda x: f"{x}_sense_rate")
    # Pivot for consolidated comments.
    comments_pivot = stats.pivot(index='candidate', columns='human', values='comments')
    comments_pivot = comments_pivot.rename(columns=lambda x: f"{x}_comments")
    
    # Combine all the pivots.
    result = pd.concat([win_rate_pivot, sense_rate_pivot, comments_pivot], axis=1).reset_index()
    
    # === Step 7. Output the results ===
    result.to_csv(output_csv, index=False)
    print(f"Output written to {output_csv}")
    return result

if __name__ == "__main__":
    # File paths (adjust as needed)
    keys_csv = "src/analysis/human/keys.csv"
    human_csv_files = ["src/analysis/human/amalie.csv", "src/analysis/human/leon.csv", "src/analysis/human/niels.csv"]
    output_csv = "model_human_stats.csv"
    
    # Compute and save the model-centered statistics.
    compute_model_human_stats(keys_csv, human_csv_files, output_csv)
