# Business Entity Resolution

Submission pipeline for the **Amazon ML Business Entity Resolution Challenge**.

## 1. Problem Overview
The goal of this challenge is to resolve records across three independent business data sources:
* **Source 1**: Reference source (deduplicated).
* **Source 2**: Target candidate source.
* **Source 3**: Target candidate source.

For every Source 1 record in the test set, the system determines which Source 2 and/or Source 3 records refer to the exact same real-world business entity. 

A Source 1 entity can have:
* Zero matches (true singleton)
* One match
* Multiple matches across S2 and S3

Matching predictions are evaluated using entity macro-averaged **$F_{0.5}$ score** (where Precision is weighted twice as heavily as Recall).

---

## 2. Environment & Requirements

### System Requirements
* Python 3.10 or higher
* Windows / Linux / macOS

### Dependencies
Dependencies are pinned in `requirements.txt`:
* `pandas`
* `numpy`
* `scikit-learn`
* `scipy`
* `rapidfuzz`
* `joblib`

### Environment Setup

#### Windows (PowerShell / CMD)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

#### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Dataset Directory Structure
The pipeline expects TSV files in the root `dataset/` directory:

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

Each source file contains tab-separated fields:
* `entity_id` (prefixed with `S1-`, `S2-`, or `S3-`)
* `business_name`
* `business_address`
* `country` (open-set, including US, India, France, etc.)

---

## 4. Pipeline Execution & Commands

### End-to-End Automated Pipeline
To execute the complete pipeline (training -> validation threshold search -> test inference -> official validation):

```bash
python code/business_entity_resolution/src/pipeline.py
```

### Individual Execution Stages

#### Stage 1: Dataset Generation (Sample Benchmark)
If challenge TSVs are not yet present:
```bash
python dataset/generate_dataset.py
```

#### Stage 2: Model Training & Validation Threshold Tuning
```bash
python -m src.train
```
*(Run from `code/business_entity_resolution` directory, or set `PYTHONPATH=code/business_entity_resolution/src`)*

#### Stage 3: Test Set Inference & Output Generation
```bash
python -m src.predict
```

#### Stage 4: Run Challenge Submission Validator
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

#### Stage 5: Run Automated Unit Tests
```bash
python -m unittest discover -s code/business_entity_resolution/tests
```

---

## 5. Generated Outputs
The pipeline generates two output TSVs in `output/`:

1. `output/matching_results.tsv`:
   Contains exact matching predictions for every Source 1 entity in `test_source1.tsv`.
   Headers: `source1_entity_id`, `matched_entity_ids` (comma-separated S2/S3 IDs, or empty string).

2. `output/candidate_pairs.tsv`:
   Contains the candidate list passed to the ML classifier for every Source 1 entity.
   Headers: `source1_entity_id`, `candidate_entity_ids` (comma-separated S2/S3 candidate IDs).

---

## 6. Reproducibility
All non-deterministic operations (GroupShuffleSplit, model initialization, blocking sampling) use a fixed seed `random_state = 42`. Running the pipeline from a fresh environment produces identical outputs and validation scores.
