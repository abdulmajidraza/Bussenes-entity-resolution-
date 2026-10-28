from evaluation import threshold_search, evaluate_predictions

def run_validation_report(val_scores_by_entity: dict, val_truth: dict):
    best_f05, best_threshold, val_metrics = threshold_search(val_scores_by_entity, val_truth)
    return {
        "macro_f05": best_f05,
        "threshold": best_threshold,
        "metrics": val_metrics
    }

if __name__ == "__main__":
    print("Validation helper loaded.")
