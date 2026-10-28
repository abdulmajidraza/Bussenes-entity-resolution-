import re
from collections import defaultdict
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio
from sklearn.feature_extraction.text import TfidfVectorizer

def _tokens(s):
    return set(s.split()) if s else set()

def build_indices(source2: pd.DataFrame, source3: pd.DataFrame):
    country_index = defaultdict(set)
    name_token_index = defaultdict(set)
    prefix_index = defaultdict(set)
    postal_index = defaultdict(set)

    targets = pd.concat([source2, source3], ignore_index=True)
    target_dict = {}

    for _, r in targets.iterrows():
        eid = r["entity_id"]
        target_dict[eid] = r

        c_norm = r["country_norm"]
        if c_norm:
            country_index[c_norm].add(eid)

        name_n = r["name_norm"]
        for t in _tokens(name_n):
            if len(t) >= 3:
                name_token_index[t].add(eid)
                prefix_index[t[:3]].add(eid)

        addr_n = r["address_norm"]
        # Numbers / postal code tokens
        nums = [t for t in addr_n.split() if t.isdigit() and len(t) >= 3]
        for n in nums:
            postal_index[n].add(eid)

    return target_dict, country_index, name_token_index, prefix_index, postal_index

def build_tfidf_retriever(source2: pd.DataFrame, source3: pd.DataFrame):
    targets = pd.concat([source2, source3], ignore_index=True)
    if len(targets) == 0:
        return None, [], None

    target_ids = targets["entity_id"].tolist()
    text_corpus = (targets["name_norm"] + " " + targets["address_norm"]).tolist()

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=1)
    tfidf_mat = vectorizer.fit_transform(text_corpus)

    return vectorizer, target_ids, tfidf_mat

def generate_candidates(source1: pd.DataFrame, source2: pd.DataFrame, source3: pd.DataFrame, top_k: int = 50, gt_dict: dict = None):
    target_dict, country_idx, name_token_idx, prefix_idx, postal_idx = build_indices(source2, source3)
    vectorizer, target_ids, tfidf_mat = build_tfidf_retriever(source2, source3)
    target_id_to_idx = {eid: idx for idx, eid in enumerate(target_ids)}

    candidate_rows = []
    total_generated_pairs = 0
    total_gt_matches = 0
    retained_gt_matches = 0

    if vectorizer is not None and len(source1) > 0:
        s1_corpus = (source1["name_norm"] + " " + source1["address_norm"]).tolist()
        s1_tfidf = vectorizer.transform(s1_corpus)
        # Sparse matrix dot product for top-K retrieval
        sim_matrix = s1_tfidf.dot(tfidf_mat.T)
    else:
        sim_matrix = None

    for i, (_, s1_row) in enumerate(source1.iterrows()):
        s1_id = s1_row["entity_id"]
        candidate_set = set()

        # Block 1: Country exact match
        c_norm = s1_row["country_norm"]
        if c_norm and c_norm in country_idx:
            candidate_set.update(country_idx[c_norm])

        # Block 2: Name token & prefix matches
        s1_name_n = s1_row["name_norm"]
        for t in _tokens(s1_name_n):
            if len(t) >= 3:
                candidate_set.update(name_token_idx.get(t, set()))
                candidate_set.update(prefix_idx.get(t[:3], set()))

        # Block 3: Postal / numeric address tokens
        s1_addr_n = s1_row["address_norm"]
        nums = [t for t in s1_addr_n.split() if t.isdigit() and len(t) >= 3]
        for n in nums:
            candidate_set.update(postal_idx.get(n, set()))

        # Block 4: TF-IDF retrieval top candidates
        if sim_matrix is not None:
            row_sims = sim_matrix[i].toarray().ravel()
            top_indices = np.argsort(row_sims)[-top_k:]
            for idx in top_indices:
                if row_sims[idx] > 0.10:
                    candidate_set.add(target_ids[idx])

        # Fast heuristic re-ranking of candidate UNION to retain top_k
        scored = []
        for eid in candidate_set:
            t_row = target_dict[eid]
            ns = ratio(s1_name_n, t_row["name_norm"]) / 100.0
            ads = ratio(s1_addr_n, t_row["address_norm"]) / 100.0
            cs = 1.0 if c_norm == t_row["country_norm"] else 0.0
            
            # Incorporate tfidf similarity if available
            tfidf_s = 0.0
            if sim_matrix is not None and eid in target_id_to_idx:
                tfidf_s = sim_matrix[i, target_id_to_idx[eid]]
                
            score = 0.45 * ns + 0.30 * ads + 0.15 * tfidf_s + 0.10 * cs
            scored.append((score, eid))

        scored.sort(reverse=True)
        chosen_candidates = [eid for _, eid in scored[:top_k]]
        candidate_rows.append((s1_id, chosen_candidates))
        total_generated_pairs += len(chosen_candidates)

        # Ground truth tracking for blocking recall
        if gt_dict and s1_id in gt_dict:
            gt_matches = set(gt_dict[s1_id])
            total_gt_matches += len(gt_matches)
            retained_gt_matches += len(gt_matches & set(chosen_candidates))

    # Blocking stats calculation
    num_s1 = len(source1)
    num_targets = len(target_dict)
    total_possible_pairs = num_s1 * num_targets
    reduction_ratio = 1.0 - (total_generated_pairs / float(total_possible_pairs)) if total_possible_pairs > 0 else 0.0
    blocking_recall = (retained_gt_matches / float(total_gt_matches)) if total_gt_matches > 0 else 1.0

    stats = {
        "num_s1": num_s1,
        "num_targets": num_targets,
        "total_possible_pairs": total_possible_pairs,
        "candidate_pairs": total_generated_pairs,
        "reduction_ratio": reduction_ratio,
        "blocking_recall": blocking_recall,
    }

    return candidate_rows, stats
