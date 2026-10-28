# Business Entity Resolution — Methodology

## 1. Methodology Used

Describe the complete entity-resolution pipeline, including preprocessing, blocking, feature engineering, model training and threshold selection.

## 2. Candidate Generation / Blocking Strategy

Describe the blocking keys and explain how candidate recall and reduction ratio were measured.

## 3. Feature Engineering

Describe business-name, address, country and numeric-token similarity features.

## 4. Model Architecture

Document the exact classifier, hyperparameters, random seed and training procedure.

## 5. Validation and F0.5 Threshold Selection

Describe the entity-level validation split and threshold search. Report only measured values from the actual run.

## 6. Error Analysis

Include examples of false positives, false negatives, difficult name/address variations and singleton handling from the validation set.

## 7. Reproducibility

Provide the exact commands and expected dataset structure required to reproduce the outputs.

## 8. Fair Play / Data Sources

State that no external business lookup, geocoding, government database, commercial entity-resolution API, web search, or external data augmentation was used.
