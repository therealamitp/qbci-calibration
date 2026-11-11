# Reproducing and extending the study

## Quick software verification

From a fresh checkout, install the package as in the README, then:

```bash
python -m unittest discover -s tests -v
python scripts/verify_circuits.py
qbci smoke --out outputs/smoke --subjects 3 --steps 3
python scripts/shot_sensitivity.py --out outputs/shots
python scripts/sequential_example.py --out outputs/replay --budget 16
python scripts/power_sensitivity.py
```

The smoke command constructs synthetic session-shifted EEG-like signals. These signals are fixtures for checking code paths, not a validated model of neurophysiology, and their ranking of methods must not motivate empirical claims or hyperparameter selection. `scripts/plot_curves.py` creates a descriptive curve plot with a synthetic warning; it is not a confirmatory uncertainty plot.

Results include five budgets per method, fixed-suffix trial predictions, target class counts, circuit/resource metadata and package versions. Output directories are not overwritten. Keep manifests, raw predictions and failures when archiving a real experiment.

## Session export

Install optional EEG dependencies. Run the data audit in [protocol.md](protocol.md). The explicit `--cue-offsets` arguments describe seconds from each loader event to cue onset, one per session. They are not analysis-window start times. Do not copy an offset from an unrelated loader version. IV-2a's inspected raw convention uses +2; OpenBMI training uses 0. IV-2b must be checked across all five sessions and feedback regimes.

The export files contain arrays plus metadata and can be loaded with `qbci.data.Session.load`. They use `allow_pickle=False`. Neither those files nor raw recordings belong in the public code repository.

## Development API

```python
from dataclasses import replace, asdict
from qbci.data import Session
from qbci.experiment import ExperimentConfig
from qbci.validation import nested_subject_evaluation, select_config

# Construct from audited earlier-to-later session paths, grouped by participant.
pairs = [(Session.load(source_path), Session.load(target_path))
         for source_path, target_path in audited_paths]
base = ExperimentConfig()
candidates = [replace(base, C=c, anchor=a)
              for c in (0.1, 1.0, 10.0) for a in (0.01, 0.1, 1.0)]
outer_results = nested_subject_evaluation(pairs, candidates, "projected_subset")
chosen, development_scores = select_config(pairs, candidates, "projected_subset")
```

This illustrates the API, not a completed search or the final fair candidate grid. Select each model family on its own relevant predeclared grid. Do not put OpenBMI pairs into this function: it rejects them. Development labels can guide development selection; held-out participant suffix labels cannot influence that fold's settings. A source-fitted personalized model for the held-out participant is allowed.

For IV-2b fixed-source experiments, pair session 1 with 2, 3, 4 and 5. Group all four under the same participant. Cross-dataset numeric participant IDs are qualified by dataset; confirm real-world independence before pooling cohorts.

## Confirmation lock

After the actual development search and dataset audit, create a lock in an ignored result directory, or archive it externally. Commit all code first. The following intentionally contains placeholders requiring real completed work:

```python
import json
from pathlib import Path
from qbci.validation import create_lock

lock = create_lock(
    configs=selected_config_dicts_by_method,
    comparator=development_selected_classical_method,
    development_subjects=audited_development_participant_ids,
    code_commit=clean_full_git_commit,
    audit_sha256=real_data_audit_sha256,
    seeds=(0, 1, 2),
)
Path("outputs/confirmation-lock.json").write_text(json.dumps(lock, indent=2))
```

Run OpenBMI with `qbci run --confirmation-lock outputs/confirmation-lock.json ...`. Method configurations, seed, code commit and clean working tree must match. Use separate invocations when methods have different selected settings. The local hash detects accidental edits; it does not authenticate authorship, enforce true temporal preregistration or prevent a person from creating a new lock after seeing outcomes. Archive the protocol and its date through an appropriate public registration service before confirmation.

## Release checks

- Run the integrity suite and executable examples after code changes.
- Inspect the versioned circuit-check artifact and actual CI result.
- Record all dataset exports, original-file checksums and environment locks before real EEG analysis.
- Keep confirmation results separate from subsequent exploratory variants.
- Publish participant-level effects and code/config revisions even when the quantum result is negative.

The package's compatible dependency ranges support installation. For a scientific run, create an exact environment lock from a successful modern environment and archive it. Do not describe the current bounds as a fully pinned reproduction environment.
