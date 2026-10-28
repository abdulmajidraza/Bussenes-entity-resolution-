# Business Entity Resolution Methodology Documentation

## 1. Executive Summary
This document presents the complete technical methodology, architecture, and empirical evaluation for the **Business Entity Resolution** solution. The system matches entity records from an authoritative reference dataset (Source 1) against records in target data sources (Source 2 and Source 3) without common cross-source identifiers.

---

## 2. Pipeline Architecture & Methodology
The solution employs a 5-stage tabular entity resolution framework:

```text
Raw TSV Datasets (Source 1, Source 2, Source 3)
                      │
                      ▼
 ┌──────────────────────────────────────────┐
 │  Stage 1: Text Normalization & Cleaning  │
 └────────────────────┬─────────────────────┘
                      │
                      ▼
 ┌──────────────────────────────────────────┐
 │ Stage 2: Multi-Strategy Candidate        │
 │          Blocking & Retrieval            │
 └────────────────────┬─────────────────────┘
                      │
                      ▼
 ┌──────────────────────────────────────────┐
 │ Stage 3: Pairwise Feature Engineering    │
 └────────────────────┬─────────────────────┘
                      │
                      ▼
 ┌──────────────────────────────────────────┐
 │ Stage 4: Tabular ML Classifier &         │
 │          Group-Split Threshold Tuning    │
 └────────────────────┬─────────────────────┘
                      │
                      ▼
 ┌──────────────────────────────────────────┐
 │ Stage 5: Test Set Inference & Safeguard  │
 │          Output Generation               │
 └──────────────────────────────────────────┘
```

---

## 3. Candidate Generation / Blocking Strategy
Comparing every Source 1 record against all Source 2/Source 3 records produces an $O(N \times M)$ search space, which is computationally expensive for large datasets. To achieve high candidate recall while dramatically reducing comparison pairs, we implement a multi-index UNION blocking strategy:

1. **Block 1 — Country Exact Match Index**: Groups target records by normalized country code.
2. **Block 2 — Name Token & 3-Gram Prefix Index**: Maps normalized name tokens (len $\ge 3$) and 3-character prefixes to target entity IDs.
3. **Block 3 — Address Postal Code & Digit Index**: Indexes sequences of numeric digits (postal codes, street numbers, building numbers).
4. **Block 4 — TF-IDF Character N-Gram Cosine Retrieval**: Builds a character 3-gram TF-IDF vector space model over combined name and address text, retrieving top candidates via sparse matrix dot product.

Candidates from all four blocks are **unioned** and re-ranked using a fast composite string heuristic ($0.45 \times \text{NameRatio} + 0.30 \times \text{AddressRatio} + 0.15 \times \text{TFIDF} + 0.10 \times \text{CountryMatch}$) to yield the final `TOP_K_PER_BLOCK` candidates per Source 1 entity.

---

## 4. Feature Engineering
For every candidate pair $(S_1, S_2/S_3)$, we construct a 24-dimensional similarity vector:

### Name Features
* **Normalized Levenshtein Ratio** (`name_ratio`)
* **RapidFuzz WRatio** (`name_wratio`)
* **Token Sort Ratio** (`name_token_sort`)
* **Token Set Ratio** (`name_token_set`)
* **Token Jaccard Similarity** (`name_jaccard`)
* **Exact Match Indicator** (`name_exact`)
* **Absolute Length Difference** (`name_len_diff`)
* **Token Count Difference** (`name_token_count_diff`)
* **Original Name Ratio** (`name_orig_ratio`)
* **Original Name Exact Indicator** (`name_orig_exact`)

### Address Features
* **Normalized Address Levenshtein Ratio** (`addr_ratio`)
* **RapidFuzz Address WRatio** (`addr_wratio`)
* **Address Token Sort Ratio** (`addr_token_sort`)
* **Address Token Set Ratio** (`addr_token_set`)
* **Address Token Jaccard** (`addr_jaccard`)
* **Address Exact Match Indicator** (`addr_exact`)
* **Address Length Difference** (`addr_len_diff`)
* **Numeric Token Overlap Count** (`addr_numeric_overlap`)
* **Numeric Token Jaccard** (`addr_numeric_jaccard`)

### Country & Meta Features
* **Country Exact Match** (`country_exact`)
* **Country Mismatch Indicator** (`country_mismatch`)
* **Field Presence Flags** (`name_present`, `address_present`)
* **Combined Similarity Score** (`combined_sim_score`)

---

## 5. Model Architecture & Validation Strategy

### Classifier
We use `HistGradientBoostingClassifier`, a gradient boosted decision tree implementation optimized for tabular data:
* `max_iter = 300`
* `learning_rate = 0.05`
* `max_leaf_nodes = 31`
* `l2_regularization = 1.0`
* `random_state = 42`

### Data Leakage Prevention
To prevent data leakage during hyperparameter and threshold tuning, training data is split into 80% train / 20% validation using **GroupShuffleSplit grouped by `source1_entity_id`**. Records belonging to the same Source 1 entity never appear in both train and validation splits.

### Validation Threshold Optimization ($F_{0.5}$)
The competition evaluation metric is macro-averaged $F_{0.5}$:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

Precision is weighted twice as heavily as recall ($\beta = 0.5$). False positives are severely penalized. Singletons (Source 1 entities with no true match) receive a score of $1.0$ if correctly predicted empty, and $0.0$ if a match is incorrectly predicted.

We perform a fine-grained grid search over probability thresholds $t \in [0.05, 0.95]$ on the validation set only, selecting the threshold that maximizes validation Macro $F_{0.5}$.

---

## 6. Error Analysis & Edge Cases

### 1. Singleton Records
* **Challenge**: Over-predicting matches on entities without true matches drops $F_{0.5}$ to $0.0$.
* **Handling**: High probability thresholds (optimized via $F_{0.5}$) prevent low-confidence candidate selections.

### 2. Legal Suffix & Format Variations
* **Challenge**: `Apex Technologies Corp` vs `Apex Tech Corporation Inc`.
* **Handling**: Token set ratio, token sort ratio, and legal suffix normalization (`pvt`, `ltd`, `inc`, `corp`, `co`, `gmbh`, `sarl`, `sa`) bridge legal formatting gaps.

### 3. Address Reordering & Landmark Differences
* **Challenge**: Reordered address tokens or missing postal codes.
* **Handling**: Numeric token overlap features and Token Jaccard isolate house/suite numbers regardless of token order.

### 4. Open-Set Country Validation
* **Challenge**: Test set contains unseen countries (e.g. France).
* **Handling**: Country is processed as an open-set string field without hardcoded country filters or country whitelist drops.

---

## 7. Instructions to Reproduce

1. **Activate Environment & Install Dependencies**:
   ```bash
   pip install -r code/business_entity_resolution/requirements.txt
   ```

2. **Execute Complete Pipeline**:
   ```bash
   python code/business_entity_resolution/src/pipeline.py
   ```

3. **Verify Output via Official Validator**:
   ```bash
   python utils/validate_submission.py \
       --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv \
       --test-dir dataset/test
   ```

4. **Run Pipeline Unit Tests**:
   ```bash
   python -m unittest discover -s code/business_entity_resolution/tests
   ```
