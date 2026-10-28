from pathlib import Path
import pandas as pd

def validate_basic_outputs(output_dir: Path, test_dir: Path) -> bool:
    output_dir = Path(output_dir)
    test_dir = Path(test_dir)

    s1 = pd.read_csv(test_dir / "test_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv(test_dir / "test_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(test_dir / "test_source3.tsv", sep="\t", dtype=str, keep_default_na=False)

    valid_targets = set(s2["entity_id"]) | set(s3["entity_id"])
    expected_s1_ids = set(s1["entity_id"])

    result_path = output_dir / "matching_results.tsv"
    cand_path = output_dir / "candidate_pairs.tsv"

    if not result_path.is_file() or not cand_path.is_file():
        print("Validation FAIL: Output TSV files missing.")
        return False

    result = pd.read_csv(result_path, sep="\t", dtype=str, keep_default_na=False)
    cand = pd.read_csv(cand_path, sep="\t", dtype=str, keep_default_na=False)

    if len(result) != len(s1):
        print(f"Validation FAIL: Expected {len(s1)} matching rows, got {len(result)}")
        return False

    if len(cand) != len(s1):
        print(f"Validation FAIL: Expected {len(s1)} candidate rows, got {len(cand)}")
        return False

    if not result["source1_entity_id"].is_unique:
        print("Validation FAIL: Duplicate source1_entity_id in matching_results.tsv")
        return False

    if set(result["source1_entity_id"]) != expected_s1_ids:
        print("Validation FAIL: Source 1 entity IDs mismatch in matching_results.tsv")
        return False

    for _, r in result.iterrows():
        raw = r["matched_entity_ids"]
        ids = [x.strip() for x in raw.split(",") if x.strip()] if raw else []
        if len(ids) != len(set(ids)):
            print(f"Validation FAIL: Duplicate IDs in matched list: {ids}")
            return False
        if not all(x in valid_targets for x in ids):
            print(f"Validation FAIL: Invalid ID in match list: {ids}")
            return False
        if any(x.startswith("S1-") for x in ids):
            print("Validation FAIL: Source 1 ID found in predicted matches.")
            return False

    cand_map = dict(zip(cand["source1_entity_id"], cand["candidate_entity_ids"]))
    for _, r in result.iterrows():
        sid = r["source1_entity_id"]
        raw_m = r["matched_entity_ids"]
        matches = set(x.strip() for x in raw_m.split(",") if x.strip()) if raw_m else set()
        
        raw_c = cand_map.get(sid, "")
        candidates = set(x.strip() for x in raw_c.split(",") if x.strip()) if raw_c else set()

        if not matches.issubset(candidates):
            print(f"Validation FAIL: Match {matches - candidates} not in candidate set for {sid}")
            return False

    print("Basic Output Validation: PASS")
    return True
