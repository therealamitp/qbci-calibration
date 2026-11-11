"""Participant-level endpoints. Seeds and sessions are never independent n."""
import numpy as np
from scipy.stats import t as student_t


def balanced_accuracy(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Balanced accuracy requires both evaluation classes")
    return float(np.mean([np.mean(prediction[y == c] == c) for c in (0, 1)]))


def calibration_auc(scores, budgets=(0, 4, 8, 16, 32)):
    """Trapezoidal area divided by 32, in balanced-accuracy units."""
    scores, budgets = np.asarray(scores, float), np.asarray(budgets, float)
    if scores.shape != budgets.shape or budgets[0] != 0 or np.any(np.diff(budgets) <= 0):
        raise ValueError("Invalid calibration curve")
    return float(np.sum(np.diff(budgets) * (scores[:-1]+scores[1:])/2) / budgets[-1])


def sustained_target(scores, budgets=(0, 4, 8, 16, 32), target=0.75):
    """First sampled budget at/above target with all larger budgets above it.

    Return None for failure, not 32 or zero. No interpolation or isotonic
    smoothing is allowed to create an unobserved threshold crossing.
    """
    for i, k in enumerate(budgets):
        if np.all(np.asarray(scores)[i:] >= target):
            return int(k)
    return None


def paired_summary(differences, seed=0, bootstrap=10000, confidence=0.95):
    d = np.asarray(differences, float)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("Supply one finite paired effect per participant")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(bootstrap, len(d)))].mean(axis=1)
    alpha = (1-confidence)/2
    se = d.std(ddof=1)/np.sqrt(len(d))
    critical = student_t.ppf(1-alpha, len(d)-1)
    return {"n_participants": len(d), "mean_difference": float(d.mean()),
            "median_difference": float(np.median(d)),
            "win_fraction": float(np.mean(d > 0)),
            "bootstrap_ci": np.quantile(means, [alpha, 1-alpha]).tolist(),
            "paired_t_ci": [float(d.mean()-critical*se), float(d.mean()+critical*se)]}


def holm(p_values):
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p)-np.arange(len(p))) * p[order])
    result = np.empty(len(p))
    result[order] = np.minimum(adjusted, 1)
    return result


def participant_auc_rows(rows):
    """Collapse budgets, then seeds/transitions within a dataset/person/method."""
    groups = {}
    for r in rows:
        key = (r["dataset"], str(r["subject"]), r["method"], r["source_session"],
               r["target_session"], r["seed"])
        group = groups.setdefault(key, {})
        if int(r["budget"]) in group:
            raise ValueError("Duplicate score row")
        group[int(r["budget"])] = float(r["balanced_accuracy"])
    collapsed = {}
    for key, curve in groups.items():
        if sorted(curve) != [0, 4, 8, 16, 32]:
            raise ValueError("Incomplete curve; do not silently drop failed budgets")
        value = calibration_auc([curve[k] for k in (0, 4, 8, 16, 32)])
        collapsed.setdefault(key[:3], []).append(value)
    return {key: float(np.mean(value)) for key, value in collapsed.items()}
