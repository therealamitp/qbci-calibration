# QBCI Calibration

**Can adapting a quantum feature map restore motor-imagery decoding across recording sessions with fewer labeled calibration trials?**

A publicly source-available, hypothesis-driven research program and executable benchmark for chronological, within-person left-versus-right motor imagery adaptation. Licensed under PolyForm Noncommercial 1.0.0. A quantum advantage is an outcome to test, not an assumption.

## Current status

The repository contains working exact statevector and finite-shot simulations, source-anchored adaptation, classical controls, chronological evaluation, participant-level statistics, leakage tests, and a literature-based research protocol. Synthetic runs verify software behavior only. **No real-EEG performance result, hardware experiment, or quantum advantage is claimed.** The MOABB export path requires installation and an audit against actual downloaded files before scientific use. Hyperparameters in `configs/development.json` are provisional, not selected by an EEG experiment.

- [Research program and decision criteria](docs/research_program.md)
- [Prior art and source audit](docs/prior_art.md)
- [Exact circuit specification](docs/circuits.md)
- [Dataset and leakage protocol](docs/protocol.md)
- [Implementation status, experiments, and resource accounting](docs/implementation.md)
- [Reproduction and release checks](docs/reproducibility.md)

## Install and test

Use Python 3.10 or newer in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
python scripts/verify_circuits.py
qbci smoke --out outputs/smoke --subjects 3 --steps 3
qbci summarize --scores outputs/smoke/scores.csv --classical rbf_subset --out outputs/smoke/paired.json
python scripts/plot_curves.py outputs/smoke/scores.csv --out outputs/smoke/curves.png
```

Run finite-shot target optimization and prediction:

```bash
qbci smoke --out outputs/shots-512 --subjects 1 --steps 3 --shots 512 --methods projected_subset fidelity_subset
```

For raw EEG or neural baselines:

```bash
python -m pip install -e '.[eeg,deep,dev]'
```

Dependencies are bounded compatibility ranges, not a scientific environment lock. Freeze the actual resolved versions before collecting confirmatory results.

## What is implemented

| Component | Implementation |
|---|---|
| EEG representation | Trial-local causal mu/beta filtering, OAS covariances, affine-invariant means, tangent vectors, source-only scaling and PCA |
| Primary quantum model | Four qubits, two reuploading blocks, nearest-neighbor CNOTs, local X/Y/Z features, RBF SVM |
| Quantum comparison | Trainable fidelity kernel with interleaved trainable gates |
| Adaptation | Regularized centered kernel-target alignment, source replay, encoding-only or full adaptation, frozen and classifier-only controls |
| Classical controls | Compressed RBF SVM, adaptive feature scales, trigonometric features, exact product-state twin, full tangent SVM, prefix covariance alignment, tangent Procrustes control, CSP/FBCSP regularized LDA |
| Neural control | Optional EEGNet-style implementation with a frozen feature extractor and constrained head fine-tuning |
| Evaluation | Budgets 0/4/8/16/32, chronological prefixes, identical suffix after trial 32, independent source restart at every budget |
| Replay | Score-before-reveal label oracle and separate sequential adaptation API |
| Statistics | Normalized calibration AUC, participant aggregation, paired bootstrap/t intervals, Holm adjustment, threshold failures |
| Audit | Per-trial predictions, environment manifests, parameter/circuit invariance checks, finite-shot and resource logs |

The default smoke run covers 20 method variants. EEGNet is optional and can be requested with `--methods eegnet_frozen eegnet_classifier`. The tangent Procrustes implementation is a transparent regularized control, **not a reproduction of the complete published TSA or RPA algorithms**. Strong published adaptation implementations remain a required benchmark extension before any superiority claim.

## Dataset workflow

Read [the dataset audit procedure](docs/protocol.md) before exporting. Explicit cue offsets are mandatory because raw event conventions differ between datasets and loader releases. For example, after verifying that the installed IV-2a loader marks trial start and that the cue follows by two seconds:

```bash
qbci export-moabb --dataset BNCI2014_001 --subject 1 --cue-offsets 2 2 --out data/iv2a
qbci run --source data/iv2a/BNCI2014_001_sub-1_session-1.npz --target data/iv2a/BNCI2014_001_sub-1_session-2.npz --config configs/development.json --out outputs/iv2a-sub-1
```

This downloads data through MOABB when the optional dependencies and network access are available. No participant EEG files are distributed here. OpenBMI uses training runs only until online-run ground-truth provenance is verified. OpenBMI experiments require a confirmation lock and a matching code revision; see [reproduction instructions](docs/reproducibility.md).

## Research position

Earlier-to-later quantum EEG classification already exists. The proposed contribution is the combination of chronological target-label budgets, representation-isolating controls, participant-level inference, and independent confirmation. A successful simulator experiment would support predictive or label-efficiency claims. It would not establish computational quantum advantage: four to eight qubits are cheaply classically simulated, and the projected model has an explicit 12-dimensional classical output at four qubits.

Please report negative results, failed threshold attainment, and favorable classical mechanisms as first-class outcomes. The proposed confirmatory minimum is **0.03 absolute normalized calibration AUC**, with uncertainty and a development-selected strongest classical comparator. This threshold is a prospective design choice, not an empirically justified clinical standard.

## License and citation

Code and original documentation are available under the [PolyForm Noncommercial License 1.0.0](LICENSE), with attribution in [NOTICE](NOTICE). Uses outside its permitted purposes require separate permission. Its permitted purposes include noncommercial use and the educational/research organization uses specified in the license. Because it restricts use, this is source-available software rather than OSI-approved open-source software. Dataset access conditions and third-party papers retain their own licenses. Use [CITATION.cff](CITATION.cff) to cite the software. This is research software and does not establish online BCI control performance.
