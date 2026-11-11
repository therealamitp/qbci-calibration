"""Chronological plans and a label oracle for score-before-reveal replay."""
from dataclasses import dataclass
import numpy as np

BUDGETS = (0, 4, 8, 16, 32)


@dataclass(frozen=True)
class PrefixPlan:
    budget: int
    calibration: np.ndarray
    unused: np.ndarray
    evaluation: np.ndarray


def prefix_plans(n_trials, budgets=BUDGETS):
    budgets = tuple(budgets)
    if budgets != tuple(sorted(set(budgets))) or budgets[0] != 0:
        raise ValueError("Budgets must be unique, increasing, and start at zero")
    maximum = max(budgets)
    if n_trials <= maximum:
        raise ValueError("No untouched suffix remains")
    return [PrefixPlan(k, np.arange(k), np.arange(k, maximum), np.arange(maximum, n_trials))
            for k in budgets]


def representation_update_allowed(labels, min_per_class=2):
    labels = np.asarray(labels, int)
    return all(np.count_nonzero(labels == c) >= min_per_class for c in (0, 1))


def source_replay_indices(y, count=32, seed=0):
    rng = np.random.default_rng(seed)
    selected = []
    for c in (0, 1):
        ids = np.flatnonzero(y == c)
        if not len(ids):
            raise ValueError("Source session must contain both classes")
        selected.extend(rng.permutation(ids)[:count//2])
    return np.sort(selected)


def domain_weights(n_source, n_target, target_mass=0.5):
    if n_source < 1:
        raise ValueError("Source replay is required")
    if n_target == 0:
        return np.ones(n_source) / n_source
    return np.r_[np.full(n_source, (1-target_mass)/n_source),
                 np.full(n_target, target_mass/n_target)]


class LabelOracle:
    """Only reveals a label after the corresponding trial has been scored."""
    def __init__(self, labels, budget):
        self._labels = np.asarray(labels, int).copy()
        self.budget = int(budget)
        self.scored = {}
        self.revealed = set()
        self.events = []

    def score(self, index, prediction):
        if index in self.scored or index != len(self.scored):
            raise ValueError("Trials must be scored exactly once in order")
        self.scored[index] = int(prediction)
        self.events.append(("score", index))

    def reveal(self, index):
        if index not in self.scored or index in self.revealed:
            raise ValueError("Score once before revealing once")
        if len(self.revealed) >= self.budget:
            raise ValueError("Label budget exhausted")
        self.revealed.add(index)
        self.events.append(("reveal", index))
        return int(self._labels[index])


def validate_pair(source, target):
    if source.subject != target.subject or source.dataset != target.dataset:
        raise ValueError("Primary transfer must be within person and dataset")
    if source.session_order >= target.session_order:
        raise ValueError("Only earlier-to-later session transfer is permitted")
    if source.channels != target.channels or source.sfreq != target.sfreq:
        raise ValueError("Channel order and sampling rate must match")
