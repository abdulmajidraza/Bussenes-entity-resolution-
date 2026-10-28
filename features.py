import re
import numpy as np
from rapidfuzz.fuzz import ratio, WRatio, token_sort_ratio, token_set_ratio

FEATURE_NAMES = [
    # Name features
    "name_ratio",
    "name_wratio",
    "name_token_sort",
    "name_token_set",
    "name_jaccard",
    "name_exact",
    "name_len_diff",
    "name_token_count_diff",
    "name_orig_ratio",
    "name_orig_exact",
    # Address features
    "addr_ratio",
    "addr_wratio",
    "addr_token_sort",
    "addr_token_set",
    "addr_jaccard",
    "addr_exact",
    "addr_len_diff",
    "addr_numeric_overlap",
    "addr_numeric_jaccard",
    # Other / composite features
    "country_exact",
    "country_mismatch",
    "name_present",
    "address_present",
    "combined_sim_score",
]

def token_jaccard(a: str, b: str) -> float:
    set_a = set(a.split()) if a else set()
    set_b = set(b.split()) if b else set()
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return float(len(set_a & set_b)) / float(len(set_a | set_b))

def numeric_tokens(s: str) -> set:
    if not s:
        return set()
    return set(re.findall(r"\b\d+\b", s))

def pair_features(a_row, b_row) -> np.ndarray:
    an = a_row.get("name_norm", "")
    bn = b_row.get("name_norm", "")
    ao = a_row.get("business_name", "")
    bo = b_row.get("business_name", "")

    aa = a_row.get("address_norm", "")
    ba = b_row.get("address_norm", "")
    
    ac = a_row.get("country_norm", "")
    bc = b_row.get("country_norm", "")

    # Name features
    nr = ratio(an, bn) / 100.0
    nwr = WRatio(an, bn) / 100.0
    nts = token_sort_ratio(an, bn) / 100.0
    ntset = token_set_ratio(an, bn) / 100.0
    nj = token_jaccard(an, bn)
    ne = float(an == bn and bool(an))
    nld = float(abs(len(an) - len(bn)))
    ntcd = float(abs(len(an.split()) - len(bn.split())))
    nor = ratio(ao, bo) / 100.0
    noe = float(ao == bo and bool(ao))

    name_feats = [nr, nwr, nts, ntset, nj, ne, nld, ntcd, nor, noe]

    # Address features
    ar = ratio(aa, ba) / 100.0
    awr = WRatio(aa, ba) / 100.0
    ats = token_sort_ratio(aa, ba) / 100.0
    atset = token_set_ratio(aa, ba) / 100.0
    aj = token_jaccard(aa, ba)
    ae = float(aa == ba and bool(aa))
    ald = float(abs(len(aa) - len(ba)))

    num_a = numeric_tokens(aa)
    num_b = numeric_tokens(ba)
    num_overlap = float(len(num_a & num_b))
    num_jaccard = float(len(num_a & num_b)) / float(len(num_a | num_b)) if (num_a or num_b) else 1.0

    addr_feats = [ar, awr, ats, atset, aj, ae, ald, num_overlap, num_jaccard]

    # Country & presence features
    ce = float(ac == bc and bool(ac))
    cm = float(ac != bc and bool(ac) and bool(bc))
    np_flag = float(bool(an) and bool(bn))
    ap_flag = float(bool(aa) and bool(ba))
    combined_score = 0.50 * nr + 0.35 * ar + 0.15 * ce

    other_feats = [ce, cm, np_flag, ap_flag, combined_score]

    features = np.array(name_feats + addr_feats + other_feats, dtype=np.float32)
    return features
