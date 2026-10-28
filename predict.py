import json
import joblib
import pandas as pd
import numpy as np

from config import TEST, OUTPUT, ARTIFACTS, TOP_K_PER_BLOCK
from data_loader import load_test
from preprocessing import enrich
from blocking import generate_candidates
from features import pair_features

def main():
    print("Loading test data...")
    s1, s2, s3 = load_test(TEST)
    s1, s2, s3 = map(enrich, (s1, s2, s3))

    model_path = ARTIFACTS / "model.joblib"
    config_path = ARTIFACTS / "config.json"

    if not model_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("Model or configuration artifact missing. Run training step first.")

    model = joblib.load(model_path)
    cfg = json.loads(config_path.read_text())
    threshold = float(cfg["threshold"])

    print(f"Loaded model. Applied validation-optimized threshold: {threshold:.4f}")

    targets_concat = pd.concat([s2, s3], ignore_index=True)
    target_dict = {r["entity_id"]: r for _, r in targets_concat.iterrows()}
    valid_target_ids = set(target_dict.keys())

    candidate_pairs, _ = generate_candidates(s1, s2, s3, top_k=TOP_K_PER_BLOCK)

    match_rows = []
    candidate_rows = []
    s1_dict = {r["entity_id"]: r for _, r in s1.iterrows()}

    matched_entity_count = 0
    empty_match_count = 0

    for sid, raw_cids in candidate_pairs:
        src_row = s1_dict[sid]
        
        # Deduplicate candidates and ensure only valid S2/S3 IDs
        valid_cids = []
        seen_cids = set()
        for cid in raw_cids:
            if cid in valid_target_ids and cid not in seen_cids and not cid.startswith("S1-"):
                seen_cids.add(cid)
                valid_cids.append(cid)

        if valid_cids:
            X_mat = np.asarray([pair_features(src_row, target_dict[cid]) for cid in valid_cids], dtype=np.float32)
            probs = model.predict_proba(X_mat)[:, 1]
            predicted_matches = [cid for cid, p in zip(valid_cids, probs) if p >= threshold]
        else:
            predicted_matches = []

        # Deduplicate predicted matches
        predicted_matches = list(dict.fromkeys(predicted_matches))

        if predicted_matches:
            matched_entity_count += 1
        else:
            empty_match_count += 1

        candidate_rows.append({
            "source1_entity_id": sid,
            "candidate_entity_ids": ",".join(valid_cids)
        })
        match_rows.append({
            "source1_entity_id": sid,
            "matched_entity_ids": ",".join(predicted_matches)
        })

    # Convert to DataFrames and guarantee exactly 1 row per test S1 entity
    match_df = pd.DataFrame(match_rows).drop_duplicates(subset=["source1_entity_id"])
    candidate_df = pd.DataFrame(candidate_rows).drop_duplicates(subset=["source1_entity_id"])

    # Enforce exact S1 coverage alignment with test_source1.tsv
    all_s1_ids = s1["entity_id"].tolist()
    match_df = match_df.set_index("source1_entity_id").reindex(all_s1_ids).fillna("").reset_index()
    candidate_df = candidate_df.set_index("source1_entity_id").reindex(all_s1_ids).fillna("").reset_index()

    # Save output TSVs
    match_path = OUTPUT / "matching_results.tsv"
    cand_path = OUTPUT / "candidate_pairs.tsv"

    match_df.to_csv(match_path, sep="\t", index=False)
    candidate_df.to_csv(cand_path, sep="\t", index=False)

    print(f"Successfully generated outputs:")
    print(f"  {match_path}: {len(match_df)} rows")
    print(f"  {cand_path}: {len(candidate_df)} rows")
    print(f"  Test Source 1 entities: {len(all_s1_ids)}")
    print(f"  Entities with predicted matches: {matched_entity_count}")
    print(f"  Entities with no matches (singletons): {empty_match_count}")

if __name__ == "__main__":
    main()
