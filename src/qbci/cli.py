"""Command line entry points for auditable experiments and summaries."""
from dataclasses import asdict
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import numpy as np
from .data import Session, synthetic_pair
from .experiment import ExperimentConfig, DEFAULT_METHODS, run_pair
from .statistics import participant_auc_rows, paired_summary
from .validation import read_lock


def write_csv(path, rows):
    if not rows:
        raise ValueError("No rows to write")
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def environment():
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "moabb", "mne", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).strip())
    except (subprocess.SubprocessError, FileNotFoundError):
        commit = None
        dirty = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "git_commit": commit, "git_dirty": dirty}


def save_run(directory, rows, predictions, resources, config, scope):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing result directory")
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(directory / "scores.csv", rows)
    write_csv(directory / "predictions.csv", predictions)
    (directory / "resources.json").write_text(json.dumps(resources, indent=2))
    manifest = {"scope": scope, "config": asdict(config), "environment": environment(),
                "empirical_quantum_advantage_claim": False,
                "finite_shot_training": "exact source initialization; finite-shot target adaptation and deployment"}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="Synthetic software verification, not EEG evidence")
    smoke.add_argument("--out", required=True)
    smoke.add_argument("--subjects", type=int, default=3)
    smoke.add_argument("--steps", type=int, default=3)
    smoke.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    smoke.add_argument("--shots", type=int)
    run = sub.add_parser("run", help="Evaluate two audited chronological session exports")
    run.add_argument("--source", required=True)
    run.add_argument("--target", required=True)
    run.add_argument("--config", default="configs/development.json")
    run.add_argument("--out", required=True)
    run.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    run.add_argument("--confirmation-lock")
    export = sub.add_parser("export-moabb", help="Explicit download/export with cue offset audit")
    export.add_argument("--dataset", required=True, choices=["BNCI2014_001", "BNCI2014_004", "Lee2019_MI"])
    export.add_argument("--subject", required=True, type=int)
    export.add_argument("--cue-offsets", required=True, nargs="+", type=float)
    export.add_argument("--out", required=True)
    summary = sub.add_parser("summarize", help="Participant-level paired effects")
    summary.add_argument("--scores", nargs="+", required=True)
    summary.add_argument("--quantum", default="projected_subset")
    summary.add_argument("--classical", required=True)
    summary.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "export-moabb":
        from .moabb_io import export_subject
        print(json.dumps(export_subject(args.dataset, args.subject, args.out, args.cue_offsets), indent=2))
        return
    if args.command == "summarize":
        rows = []
        for path in args.scores:
            with open(path) as file:
                rows.extend(csv.DictReader(file))
        aucs = participant_auc_rows(rows)
        output = {}
        for dataset in sorted(set(k[0] for k in aucs)):
            people = sorted(set(k[1] for k in aucs if k[0] == dataset))
            d = [aucs[(dataset, p, args.quantum)]-aucs[(dataset, p, args.classical)] for p in people]
            output[dataset] = {"contrast": args.quantum+" minus "+args.classical,
                               "participant_effects": dict(zip(people, d)), **paired_summary(d)}
        Path(args.out).write_text(json.dumps(output, indent=2))
        print(json.dumps(output, indent=2))
        return
    if args.command == "smoke":
        config = ExperimentConfig(source_steps=args.steps, target_steps=args.steps, shots=args.shots)
        pairs = [synthetic_pair(seed=i, trials=64) for i in range(args.subjects)]
        scope = "SYNTHETIC SOFTWARE VERIFICATION ONLY"
    else:
        config = ExperimentConfig(**json.loads(Path(args.config).read_text()))
        pairs = [(Session.load(args.source), Session.load(args.target))]
        scope = "development"
        if pairs[0][0].dataset == "Lee2019_MI":
            if not args.confirmation_lock:
                raise ValueError("OpenBMI requires a frozen confirmation lock")
            lock = read_lock(args.confirmation_lock)
            for method in args.methods:
                expected = lock["protocol"]["configs"].get(method, {}).copy()
                expected["seed"] = config.seed
                if expected != asdict(config) or config.seed not in lock["protocol"]["seeds"]:
                    raise ValueError("Method/config does not match confirmation lock")
            actual = environment()
            if actual["git_commit"] != lock["protocol"]["code_commit"] or actual["git_dirty"]:
                raise ValueError("Clean code commit must match the confirmation lock")
            scope = "locked OpenBMI confirmation"
    scores, predictions, resources = [], [], []
    for source, target in pairs:
        print("Running %s participant %s, %s -> %s" %
              (source.dataset, source.subject, source.session, target.session), flush=True)
        rows, pred, costs = run_pair(source, target, config, args.methods)
        scores.extend(rows); predictions.extend(pred); resources.extend(costs)
    save_run(args.out, scores, predictions, resources, config, scope)
    print("Saved %d score rows; scope: %s" % (len(scores), scope))


if __name__ == "__main__":
    main()
