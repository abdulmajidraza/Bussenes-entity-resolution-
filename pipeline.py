import sys
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASET_GEN = ROOT / "dataset" / "generate_dataset.py"
TRAIN_DIR = ROOT / "dataset" / "train"
VALIDATOR = ROOT / "utils" / "validate_submission.py"
OUTPUT_DIR = ROOT / "output"
TEST_DIR = ROOT / "dataset" / "test"

def run_script(path: Path, name: str):
    print(f"\n==================================================")
    print(f"   EXECUTING STAGE: {name}")
    print(f"==================================================")
    cmd = [sys.executable, str(path)]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"Stage '{name}' failed with exit code {res.returncode}")

def run_validator():
    print(f"\n==================================================")
    print(f"   EXECUTING STAGE: Challenge Validator")
    print(f"==================================================")
    cmd = [
        sys.executable,
        str(VALIDATOR),
        "--matching", str(OUTPUT_DIR / "matching_results.tsv"),
        "--candidate", str(OUTPUT_DIR / "candidate_pairs.tsv"),
        "--test-dir", str(TEST_DIR),
    ]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"Validator reported errors! Exit code {res.returncode}")

def main():
    # Ensure dataset exists
    if not (TRAIN_DIR / "train_source1.tsv").is_file():
        print("Dataset files not detected in dataset/train. Generating benchmark dataset...")
        run_script(DATASET_GEN, "Dataset Generation")

    # Step 1: Train & Validate Threshold
    run_script(HERE / "train.py", "Model Training & Validation Optimization")

    # Step 2: Test Inference & Output Generation
    run_script(HERE / "predict.py", "Test Inference & TSV Output Generation")

    # Step 3: Run Challenge Validator
    run_validator()

    print("\n" + "*" * 60)
    print("  ALL PIPELINE STAGES COMPLETED SUCCESSFULLY AND VALIDATED!")
    print("*" * 60)

if __name__ == "__main__":
    main()
