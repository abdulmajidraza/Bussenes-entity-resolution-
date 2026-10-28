import numpy as np

def fbeta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision <= 0 or recall <= 0:
        return 0.0
    b2 = beta * beta
    return (1.0 + b2) * precision * recall / (b2 * precision + recall)

def entity_f05(predicted_ids: list, ground_truth_ids: list) -> float:
    p_set = set(predicted_ids)
    t_set = set(ground_truth_ids)

    # True singleton case
    if not t_set:
        return 1.0 if not p_set else 0.0

    # Non-singleton ground truth but predicted empty
    if not p_set:
        return 0.0

    tp = len(p_set & t_set)
    precision = tp / float(len(p_set))
    recall = tp / float(len(t_set))
    return fbeta(precision, recall, beta=0.5)

def evaluate_predictions(pred_dict: dict, truth_dict: dict) -> dict:
    f05_scores = []
    precisions = []
    recalls = []
    tp_total = 0
    fp_total = 0
    fn_total = 0
    correct_singletons = 0
    actual_singletons = 0
    predicted_singletons = 0

    for eid, actual in truth_dict.items():
        predicted = pred_dict.get(eid, [])
        score = entity_f05(predicted, actual)
        f05_scores.append(score)

        p_set = set(predicted)
        t_set = set(actual)

        if not t_set:
            actual_singletons += 1
        if not p_set:
            predicted_singletons += 1
        if not t_set and not p_set:
            correct_singletons += 1

        if p_set and t_set:
            tp = len(p_set & t_set)
            fp = len(p_set - t_set)
            fn = len(t_set - p_set)
            tp_total += tp
            fp_total += fp
            fn_total += fn
            p_val = tp / float(len(p_set))
            r_val = tp / float(len(t_set))
            precisions.append(p_val)
            recalls.append(r_val)
        elif not t_set and p_set:
            fp_total += len(p_set)
            precisions.append(0.0)
        elif t_set and not p_set:
            fn_total += len(t_set)
            recalls.append(0.0)

    macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
    macro_precision = float(np.mean(precisions)) if precisions else (1.0 if not fp_total else 0.0)
    macro_recall = float(np.mean(recalls)) if recalls else (1.0 if not fn_total else 0.0)

    return {
        "macro_f05": macro_f05,
        "precision": macro_precision,
        "recall": macro_recall,
        "tp_total": tp_total,
        "fp_total": fp_total,
        "fn_total": fn_total,
        "actual_singletons": actual_singletons,
        "predicted_singletons": predicted_singletons,
        "correct_singletons": correct_singletons,
    }

def threshold_search(scores_by_entity: dict, truth_dict: dict, grid_step: float = 0.01) -> tuple:
    best_score = -1.0
    best_threshold = 0.50
    best_metrics = {}

    thresholds = np.arange(0.05, 0.96, grid_step)
    for t in thresholds:
        t_val = round(float(t), 4)
        pred_dict = {
            eid: [cand for cand, prob in pairs if prob >= t_val]
            for eid, pairs in scores_by_entity.items()
        }
        metrics = evaluate_predictions(pred_dict, truth_dict)
        score = metrics["macro_f05"]
        if score > best_score:
            best_score = score
            best_threshold = t_val
            best_metrics = metrics

    return best_score, best_threshold, best_metrics
