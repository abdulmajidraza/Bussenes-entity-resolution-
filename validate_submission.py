import argparse
import sys
from pathlib import Path
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Validate challenge submission outputs.")
    parser.add_argument("--matching", required=True, type=str, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, type=str, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, type=str, help="Path to dataset/test directory")
    return parser.parse_args()

def validate(matching_path, candidate_path, test_dir):
    errors = []
    
    matching_p = Path(matching_path)
    candidate_p = Path(candidate_path)
    test_d = Path(test_dir)

    if not matching_p.is_file():
        errors.append(f"Matching results file not found: {matching_p}")
    if not candidate_p.is_file():
        errors.append(f"Candidate pairs file not found: {candidate_p}")
    
    s1_path = test_d / "test_source1.tsv"
    s2_path = test_d / "test_source2.tsv"
    s3_path = test_d / "test_source3.tsv"

    for p in (s1_path, s2_path, s3_path):
        if not p.is_file():
            errors.append(f"Required test file not found: {p}")

    if errors:
        for err in errors:
            print(f"[FAIL] {err}")
        return False

    # Load TSVs
    try:
        s1_df = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
        s2_df = pd.read_csv(s2_path, sep="\t", dtype=str, keep_default_na=False)
        s3_df = pd.read_csv(s3_path, sep="\t", dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"[FAIL] Error reading test source TSV files: {e}")
        return False

    valid_targets = set(s2_df["entity_id"]) | set(s3_df["entity_id"])
    expected_s1_ids = set(s1_df["entity_id"])

    try:
        match_df = pd.read_csv(matching_p, sep="\t", dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"[FAIL] Error reading matching_results.tsv: {e}")
        return False

    try:
        cand_df = pd.read_csv(candidate_p, sep="\t", dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"[FAIL] Error reading candidate_pairs.tsv: {e}")
        return False

    # 1. Column headers
    if list(match_df.columns) != ["source1_entity_id", "matched_entity_ids"]:
        errors.append(f"matching_results.tsv headers must be ['source1_entity_id', 'matched_entity_ids'], got {list(match_df.columns)}")
    if list(cand_df.columns) != ["source1_entity_id", "candidate_entity_ids"]:
        errors.append(f"candidate_pairs.tsv headers must be ['source1_entity_id', 'candidate_entity_ids'], got {list(cand_df.columns)}")

    if errors:
        for err in errors:
            print(f"[FAIL] {err}")
        return False

    # 2. Row completeness and uniqueness
    match_s1_ids = match_df["source1_entity_id"].tolist()
    cand_s1_ids = cand_df["source1_entity_id"].tolist()

    if len(match_s1_ids) != len(set(match_s1_ids)):
        errors.append("matching_results.tsv contains duplicate source1_entity_id entries.")
    if len(cand_s1_ids) != len(set(cand_s1_ids)):
        errors.append("candidate_pairs.tsv contains duplicate source1_entity_id entries.")

    if set(match_s1_ids) != expected_s1_ids:
        missing = expected_s1_ids - set(match_s1_ids)
        extra = set(match_s1_ids) - expected_s1_ids
        errors.append(f"matching_results.tsv S1 ID mismatch. Missing: {len(missing)}, Extra: {len(extra)}")

    if set(cand_s1_ids) != expected_s1_ids:
        missing = expected_s1_ids - set(cand_s1_ids)
        extra = set(cand_s1_ids) - expected_s1_ids
        errors.append(f"candidate_pairs.tsv S1 ID mismatch. Missing: {len(missing)}, Extra: {len(extra)}")

    if errors:
        for err in errors:
            print(f"[FAIL] {err}")
        return False

    # 3. Match format & candidate subset checks
    cand_map = dict(zip(cand_df["source1_entity_id"], cand_df["candidate_entity_ids"]))

    for idx, row in match_df.iterrows():
        s1_id = row["source1_entity_id"]
        raw_matches = row["matched_entity_ids"]
        matches = [x.strip() for x in raw_matches.split(",") if x.strip()] if raw_matches else []

        raw_cands = cand_map.get(s1_id, "")
        candidates = [x.strip() for x in raw_cands.split(",") if x.strip()] if raw_cands else []

        # Check duplicates
        if len(matches) != len(set(matches)):
            errors.append(f"Entity {s1_id} in matching_results.tsv has duplicate matched IDs: {matches}")
        if len(candidates) != len(set(candidates)):
            errors.append(f"Entity {s1_id} in candidate_pairs.tsv has duplicate candidate IDs: {candidates}")

        # Check invalid IDs & S1 self-matching
        for m in matches:
            if m.startswith("S1-"):
                errors.append(f"Entity {s1_id} matched to S1 ID: {m}")
            if m not in valid_targets:
                errors.append(f"Entity {s1_id} matched to unknown ID not in test_source2/3: {m}")

        # Check candidate subset rule
        matches_set = set(matches)
        cands_set = set(candidates)
        if not matches_set.issubset(cands_set):
            diff = matches_set - cands_set
            errors.append(f"Entity {s1_id} predicted matches {diff} not in candidate list")

    if errors:
        for err in errors[:20]:  # Limit output length
            print(f"[FAIL] {err}")
        if len(errors) > 20:
            print(f"[FAIL] ... and {len(errors) - 20} more errors.")
        return False

    print("PASS")
    return True

if __name__ == "__main__":
    args = parse_args()
    success = validate(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if success else 1)
