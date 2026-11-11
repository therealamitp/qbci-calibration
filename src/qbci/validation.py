"""Subject-disjoint development selection and immutable confirmation locks."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from .experiment import ExperimentConfig, run_pair
from .statistics import participant_auc_rows


def validate_development_pairs(pairs):
    for source, target in pairs:
        if "Lee2019" in source.dataset or "OpenBMI" in source.dataset:
            raise ValueError("OpenBMI is reserved for locked confirmation, never development")


def select_config(pairs, candidates, method):
    """Select using only the supplied DEVELOPMENT participants.

    This function intentionally has no optional evaluation-label argument.
    Scores average transitions within participant before averaging people.
    """
    validate_development_pairs(pairs)
    trials = []
    for config in candidates:
        rows = []
        for source, target in pairs:
            rows.extend(run_pair(source, target, config, methods=[method])[0])
        per = participant_auc_rows(rows)
        trials.append(float(np.mean(list(per.values()))))
    return candidates[int(np.argmax(trials))], trials


def nested_subject_evaluation(pairs, candidates, method):
    """Outer leave-one-person-out, inner selection on all other people.

    Dataset-qualified participant IDs avoid accidental numerical-ID collisions.
    If a person appears in both cohorts, supply a harmonized identity before use.
    """
    validate_development_pairs(pairs)
    people = sorted(set((s.dataset, s.subject) for s, _ in pairs))
    if len(people) < 3:
        raise ValueError("At least three development participants are required")
    out = []
    for person in people:
        training = [(s, t) for s, t in pairs if (s.dataset, s.subject) != person]
        testing = [(s, t) for s, t in pairs if (s.dataset, s.subject) == person]
        chosen, scores = select_config(training, candidates, method)
        for source, target in testing:
            rows = run_pair(source, target, chosen, methods=[method])[0]
            out.append({"held_out_participant": person, "config": asdict(chosen),
                        "inner_scores": scores, "rows": rows})
    return out


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def create_lock(configs, comparator, development_subjects, code_commit, audit_sha256, seeds=(0, 1, 2)):
    if not comparator or not code_commit or not audit_sha256 or not development_subjects:
        raise ValueError("A real code commit, data audit, comparator and cohort are required")
    if comparator not in configs or not seeds:
        raise ValueError("Comparator configuration and seed list are required")
    body = {"schema": 1, "configs": configs, "primary_comparator": comparator, "seeds": list(seeds),
            "development_subjects": development_subjects, "code_commit": code_commit,
            "data_audit_sha256": audit_sha256, "budgets": [0, 4, 8, 16, 32],
            "minimum_practical_auc_gain": 0.03, "performance_target": 0.75,
            "confirmation_dataset": "Lee2019_MI", "created_before_confirmation": True}
    return {"protocol": body, "sha256": canonical_hash(body)}


def read_lock(path):
    lock = json.loads(Path(path).read_text())
    if canonical_hash(lock["protocol"]) != lock["sha256"]:
        raise ValueError("Confirmation lock has changed")
    return lock
