import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit

from config import TRAIN, ARTIFACTS, RANDOM_STATE, TOP_K_PER_BLOCK
from data_loader import load_training
from preprocessing import enrich
from blocking import generate_candidates
from features import pair_features, FEATURE_NAMES
from evaluation import threshold_search, evaluate_predictions

def parse_ground_truth(gt_df: pd.DataFrame) -> dict:
    gt_dict = {}
    for _, r in gt_df.iterrows():
        s1_id = r["source1_entity_id"]
        raw = r.get("matched_entity_ids", "")
        matches = [x.strip() for x in raw.split(",") if x.strip()] if raw else []
        gt_dict[s1_id] = matches
    return gt_dict

def main():
    print("Loading training data...")
    s1, s2, s3, gt = load_training(TRAIN)
    s1, s2, s3 = map(enrich, (s1, s2, s3))
    truth = parse_ground_truth(gt)

    print("Generating candidate pairs...")
    candidate_rows, blocking_stats = generate_candidates(s1, s2, s3, top_k=TOP_K_PER_BLOCK, gt_dict=truth)

    print(f"Blocking Statistics:")
    print(f"  Total S1 entities: {blocking_stats['num_s1']}")
    print(f"  Target entities (S2+S3): {blocking_stats['num_targets']}")
    print(f"  Total possible pairs: {blocking_stats['total_possible_pairs']}")
    print(f"  Generated candidate pairs: {blocking_stats['candidate_pairs']}")
    print(f"  Reduction ratio: {blocking_stats['reduction_ratio']:.6f}")
    print(f"  Blocking recall: {blocking_stats['blocking_recall']:.4f}")

    target_dict = {r["entity_id"]: r for _, r in pd.concat([s2, s3], ignore_index=True).iterrows()}
    s1_dict = {r["entity_id"]: r for _, r in s1.iterrows()}

    X, y, groups = [], [], []
    scores_by_entity = {}

    for sid, c_ids in candidate_rows:
        actual_matches = set(truth.get(sid, []))
        src_row = s1_dict[sid]
        
        pair_list = []
        for eid in c_ids:
            tgt_row = target_dict[eid]
            feat = pair_features(src_row, tgt_row)
            is_match = 1 if eid in actual_matches else 0
            
            X.append(feat)
            y.append(is_match)
            groups.append(sid)
            pair_list.append(eid)
            
        scores_by_entity[sid] = pair_list

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    groups = np.asarray(groups)

    if len(X) == 0 or len(np.unique(y)) < 2:
        raise RuntimeError("Candidate generation produced insufficient class balance. Check blocking strategy.")

    # 80/20 GroupShuffleSplit by Source 1 entity ID to prevent data leakage
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))

    print(f"Train candidate pairs: {len(train_idx)}, Validation candidate pairs: {len(val_idx)}")
    val_s1_ids = set(groups[val_idx])

    # Fit ML tabular classifier
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )
    model.fit(X[train_idx], y[train_idx])

    # Perform validation threshold search strictly on validation set
    val_scores_by_entity = {}
    for sid in val_s1_ids:
        c_ids = scores_by_entity[sid]
        if not c_ids:
            val_scores_by_entity[sid] = []
            continue
        src_row = s1_dict[sid]
        mat = np.asarray([pair_features(src_row, target_dict[eid]) for eid in c_ids], dtype=np.float32)
        probs = model.predict_proba(mat)[:, 1]
        val_scores_by_entity[sid] = list(zip(c_ids, probs))

    val_truth = {sid: truth[sid] for sid in val_s1_ids}
    best_f05, best_threshold, val_metrics = threshold_search(val_scores_by_entity, val_truth)

    print("\n" + "=" * 50)
    print("VALIDATION REPORT")
    print("=" * 50)
    print(f"Validation entities: {len(val_s1_ids)}")
    print(f"Validation Candidate Recall: {blocking_stats['blocking_recall']:.4f}")
    print(f"Optimal Threshold: {best_threshold:.4f}")
    print(f"Validation Macro F0.5: {best_f05:.4f}")
    print(f"Validation Precision: {val_metrics['precision']:.4f}")
    print(f"Validation Recall: {val_metrics['recall']:.4f}")
    print(f"True Positives (TP): {val_metrics['tp_total']}")
    print(f"False Positives (FP): {val_metrics['fp_total']}")
    print(f"False Negatives (FN): {val_metrics['fn_total']}")
    print(f"Actual Singletons: {val_metrics['actual_singletons']}")
    print(f"Predicted Singletons: {val_metrics['predicted_singletons']}")
    print(f"Correct Singletons: {val_metrics['correct_singletons']}")
    print("=" * 50 + "\n")

    # Save trained model & metadata
    joblib.dump(model, ARTIFACTS / "model.joblib")
    (ARTIFACTS / "feature_names.json").write_text(json.dumps(FEATURE_NAMES, indent=2))
    
    config_data = {
        "threshold": best_threshold,
        "validation_f05": best_f05,
        "precision": val_metrics["precision"],
        "recall": val_metrics["recall"],
        "blocking_recall": blocking_stats["blocking_recall"],
        "top_k": TOP_K_PER_BLOCK,
        "random_state": RANDOM_STATE,
    }
    (ARTIFACTS / "config.json").write_text(json.dumps(config_data, indent=2))
    print(f"Artifacts successfully saved to {ARTIFACTS}")

if __name__ == "__main__":
    main()
