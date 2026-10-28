# Business Entity Resolution Challenge

End-to-end ML pipeline for matching Source 1 business entities to matching records in Source 2 and Source 3.

## Expected dataset

Place the challenge files here:

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

All files are TSV and must be read with `sep="\t"`.

## Pipeline

1. Load and validate data
2. Normalize business names and addresses
3. Generate candidates with multiple blocking strategies
4. Build name/address similarity features
5. Train a binary matching model
6. Tune the decision threshold using validation F0.5
7. Run inference on the full test set
8. Generate `output/matching_results.tsv` and `output/candidate_pairs.tsv`
9. Run the official challenge validator

## Run

From the project root:

```bash
python -m pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/train.py
python code/business_entity_resolution/src/predict.py
```

For a clean end-to-end run:

```bash
python code/business_entity_resolution/src/pipeline.py
```

The scripts use only the supplied challenge data. No external business lookup, geocoding, web search, or external data augmentation is used.

## Outputs

```text
output/matching_results.tsv
output/candidate_pairs.tsv
```

The final challenge package should additionally contain the filled methodology document and the runnable code package required by the problem statement.
