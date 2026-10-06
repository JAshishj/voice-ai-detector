"""Pure classification metrics (no sklearn dependency).

Shared by training/train.py and training/evaluate.py so serving thresholds
and reported numbers always use the same definitions.
"""


def f1_score(targets, preds) -> float:
    tp = sum(1 for p, t in zip(preds, targets) if p == 1 and int(t) == 1)
    fp = sum(1 for p, t in zip(preds, targets) if p == 1 and int(t) == 0)
    fn = sum(1 for p, t in zip(preds, targets) if p == 0 and int(t) == 1)
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def pr_at_threshold(targets, scores, threshold: float):
    tp = fp = tn = fn = 0
    for s, y in zip(scores, targets):
        p = 1 if s >= threshold else 0
        y = int(y)
        if p == 1 and y == 1:
            tp += 1
        elif p == 1:
            fp += 1
        elif y == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    acc = (tp + tn) / len(targets) if targets else 0.0
    return {"precision": precision, "recall": recall, "fpr": fpr, "accuracy": acc}


def auc_score(targets, scores) -> float:
    """Mann-Whitney rank AUC with average ranks for tied scores."""
    if not scores:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2  # 1-based average of tied positions
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    n_pos = sum(1 for t in targets if int(t) == 1)
    n_neg = len(targets) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = sum(r for r, t in zip(ranks, targets) if int(t) == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def eer(targets, scores) -> tuple[float, float]:
    """Equal-error-rate and its threshold (linear sweep over score deciles)."""
    if not scores:
        return 0.5, 0.5
    uniq = sorted(set(scores))
    if len(uniq) > 200:
        step = len(uniq) / 200
        uniq = [uniq[int(i * step)] for i in range(200)]
    best_t, best_gap = 0.5, float("inf")
    best_eer = 0.5
    for t in uniq:
        m = pr_at_threshold(targets, scores, t)
        fnr = 1 - m["recall"]
        gap = abs(m["fpr"] - fnr)
        if gap < best_gap:
            best_gap = gap
            best_t = t
            best_eer = (m["fpr"] + fnr) / 2
    return float(best_eer), float(best_t)


def best_threshold(targets, scores) -> float:
    """Youden's J threshold (maximizes sensitivity + specificity - 1)."""
    if not scores:
        return 0.5
    uniq = sorted(set(scores))
    if len(uniq) > 200:
        step = len(uniq) / 200
        uniq = [uniq[int(i * step)] for i in range(200)]
    best_t, best_j = 0.5, -1.0
    for t in uniq:
        m = pr_at_threshold(targets, scores, t)
        j = m["recall"] + (1 - m["fpr"]) - 1
        if j > best_j:
            best_j, best_t = j, t
    return float(best_t)


def expected_calibration_error(targets, scores, n_bins: int = 10) -> float:
    """ECE over confidence of the predicted class."""
    if not scores:
        return 0.0
    ece, n = 0.0, len(scores)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, s in enumerate(scores) if (s > lo or (b == 0 and s == lo)) and s <= hi]
        if not idx:
            continue
        conf = sum(max(s, 1 - s) for s in (scores[i] for i in idx)) / len(idx)
        acc = sum(
            1 for i in idx if (1 if scores[i] >= 0.5 else 0) == int(targets[i])
        ) / len(idx)
        ece += len(idx) / n * abs(acc - conf)
    return ece
