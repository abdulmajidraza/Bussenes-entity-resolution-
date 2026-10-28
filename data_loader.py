from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {"entity_id", "business_name", "business_address", "country"}

def load_source(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Source file not found: {path}")
    
    # Must use pd.read_csv(path, sep="\t") as required by problem statement
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path}: Missing required columns: {sorted(missing)}")
    
    return df

def load_training(train_dir: Path):
    train_dir = Path(train_dir)
    s1 = load_source(train_dir / "train_source1.tsv")
    s2 = load_source(train_dir / "train_source2.tsv")
    s3 = load_source(train_dir / "train_source3.tsv")
    
    gt_path = train_dir / "train_ground_truth.tsv"
    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    
    req_gt = {"source1_entity_id", "matched_entity_ids"}
    missing_gt = req_gt - set(gt.columns)
    if missing_gt:
        raise ValueError(f"{gt_path}: Missing required columns: {sorted(missing_gt)}")
        
    return s1, s2, s3, gt

def load_test(test_dir: Path):
    test_dir = Path(test_dir)
    s1 = load_source(test_dir / "test_source1.tsv")
    s2 = load_source(test_dir / "test_source2.tsv")
    s3 = load_source(test_dir / "test_source3.tsv")
    return s1, s2, s3
