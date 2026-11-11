"""Demonstrate a prediction-before-label ledger on synthetic EEG."""
import argparse
import json
from pathlib import Path
from qbci.data import synthetic_pair
from qbci.preprocessing import covariance_epochs, SourceFeatures
from qbci.adaptation import KernelAdapter
from qbci.replay import sequential_replay
from qbci.cli import write_csv

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", default="outputs/replay")
parser.add_argument("--budget", type=int, default=16)
args = parser.parse_args()
directory = Path(args.out)
if directory.exists() and any(directory.iterdir()):
    raise FileExistsError("Refusing to overwrite results")
directory.mkdir(parents=True, exist_ok=True)
source, target = synthetic_pair(trials=64)
cs = covariance_epochs(source.epochs, source.sfreq)
ct = covariance_epochs(target.epochs, target.sfreq)
basis = SourceFeatures(4).fit(cs)
model = KernelAdapter(source_steps=3, target_steps=3).fit_source(basis.transform(cs), source.labels)
rows, ledger = sequential_replay(model, basis.transform(ct), target.labels, budget=args.budget)
write_csv(directory / "predictions.csv", rows)
(directory / "ledger.json").write_text(json.dumps({"scope": "SYNTHETIC SOFTWARE EXAMPLE",
    "budget": args.budget, "events": ledger}, indent=2))
print("Saved %d predictions, %d revealed labels" % (len(rows), args.budget))
