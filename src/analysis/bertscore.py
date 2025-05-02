from evaluate import load
import json
import numpy as np

bertscore = load("bertscore")

with open("", "r") as f_in:
    avg_results = []
    for line in f_in:
        data = json.loads(line)
        preds = [str(list(error.values())[0]) for error in data["conversation"][0]["error"]]
        refs = [str(list(feedback.values())[0]) for feedback in data["conversation"][1]["feedback"]]
        results = bertscore.compute(predictions=preds, references=refs, model_type="roberta-large")
        avg_results.extend(results["f1"])
    
    print(np.mean(avg_results))