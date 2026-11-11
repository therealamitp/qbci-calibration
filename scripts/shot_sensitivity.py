"""Reproducible synthetic shot smoke tests, never empirical EEG evidence."""
import argparse
from dataclasses import replace
from pathlib import Path
from qbci.data import synthetic_pair
from qbci.experiment import ExperimentConfig, run_pair
from qbci.cli import save_run

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", default="outputs/shot-sensitivity")
parser.add_argument("--steps", type=int, default=3)
args = parser.parse_args()
source, target = synthetic_pair(trials=64)
base = ExperimentConfig(source_steps=args.steps, target_steps=args.steps)
for shots in (None, 128, 512, 2048):
    config = replace(base, shots=shots)
    rows, predictions, resources = run_pair(source, target, config,
        methods=["projected_classifier", "projected_subset", "fidelity_classifier", "fidelity_subset"])
    save_run(Path(args.out) / str(shots), rows, predictions, resources, config,
             "SYNTHETIC SHOT SOFTWARE VERIFICATION ONLY")
    print("Completed shots=%s: %d rows" % (shots, len(rows)), flush=True)
