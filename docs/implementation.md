# Implementation coverage and computational accounting

## Delivered modules

| File | Function and scope |
|---|---|
| `src/qbci/quantum.py` | Batched NumPy statevectors, explicit rotations/CNOTs, local XYZ, joint-basis finite-shot sampling, fidelity kernels and exact product-state twin |
| `src/qbci/kernels.py` | RBF, source bandwidth, weighted centered alignment, train-only PSD repair and kernel diagnostics |
| `src/qbci/adaptation.py` | Source initialization, source-anchored subset/full SPSA adaptation, classifier refitting, trigonometric/RBF controls, resource counters |
| `src/qbci/preprocessing.py` | Causal filtering, OAS covariance, SPD functions, Riemannian mean/geodesic, tangent representation, frozen source PCA/scalers |
| `src/qbci/classical.py` | Weighted shrinkage LDA, source-fitted CSP/FBCSP, replay-weighted SVM, regularized centroid Procrustes |
| `src/qbci/eegnet.py` | Optional independent EEGNet-style baseline, source standardization, constrained head fine-tuning and frozen BN statistics |
| `src/qbci/protocol.py` | Chronological prefix plans, common suffix, source replay, domain weights, session identity checks, score-before-label oracle |
| `src/qbci/experiment.py` | Primary runner for 20 default variants and two optional EEGNet variants |
| `src/qbci/replay.py` | Separate sequential score/reveal/update evaluation |
| `src/qbci/data.py`, `moabb_io.py` | Safe numeric session serialization, synthetic fixture, explicit-cue-offset MOABB exporter and release audit |
| `src/qbci/statistics.py` | Balanced accuracy, normalized AUC, sustained threshold attainment, participant pairing, intervals and Holm adjustment |
| `src/qbci/validation.py` | Development-only selection, outer participant holdout, confirmation configuration hash |
| `src/qbci/mechanisms.py` | Class separation, source-target MMD and finite-difference parameter sensitivity |
| `src/qbci/cli.py` | Synthetic runs, audited session-pair runs, export, participant-level summaries and manifests |

The source models and controls are executable implementations, not placeholder class definitions. The report intentionally distinguishes them from mature published methods they only resemble.

## Executed versus pending

Executed locally: the core unit/integration suite, including optional PyTorch fine-tuning tests; exact circuit probes; synthetic default-method calibration runs; and synthetic finite-shot checks. The release verification artifact records the actual commands and results. Unit tests check malformed splits, chronological label access, one-class behavior, data identity, prefix isolation, PSD treatment and circuit behavior, not merely import success.

Not executed: original EEG downloads/exports, development-cohort model selection, all-participant experiments, OpenBMI confirmation, a device-noise model, hardware jobs, or online BCI control. No empirical superiority finding is implied by the synthetic runs.

MOABB/MNE are optional and were unavailable in the execution host; dependency installation was blocked by network name resolution. The exporter was reviewed against online loader source but has not been integration-tested against downloaded EEG. The host's available Python 3.8 scientific stack ran the core tests. The supported installation target is Python 3.10+; GitHub Actions is configured to test 3.10 and 3.12. A passing host test is not evidence that a remote CI run passed; inspect Actions separately.

Remaining work required before a strong benchmark claim:

- Pin and audit actual data releases, native files and cue timing; validate all MOABB exporter paths.
- Reproduce complete published RPA/TSA, rather than labeling the regularized centroid control as equivalent.
- Expose and fairly tune baseline-specific regularization, alignment, spatial filters and neural optimizer settings.
- Implement the shared-prefix-alignment factorial arm and a complete mechanistic reporting pipeline.
- Execute subject-disjoint selection with logged candidate failures and a locked seed list.
- Archive the confirmation protocol outside this repository before viewing OpenBMI outcomes.
- Benchmark memory, preprocessing, training, optimization and inference on declared hardware; evaluate device noise before hardware claims.

## Resource model

The log separates exact simulation statevectors, objective calls, logical circuit settings, shots, state-preparation one-/two-qubit gates, measurement-basis gates, Gram entries and wall time. Source training is reported separately from target adaptation/prediction. Shared source caches are marked; do not sum cached source costs as though each target budget required retraining, and do not omit source costs from an end-to-end deployment comparison.

`source_preprocessing_seconds` currently covers the cached source and target per-trial covariance construction plus fitting the source representation. It is a shared run setup cost, despite the historical field name. `source_setup_seconds` covers each method's additional setup; a cache hit is identified. `seconds` is budget-specific adaptation plus suffix prediction. There is no hardware queue/transpilation latency in a simulator measurement. Memory usage and framework compilation still require dedicated instrumentation.

For projected kernels with N inputs and S shots per global Pauli setting, feature construction requires 3NS preparations and 3N settings. The classical RBF Gram still costs O(N^2 q). When parameters change, all source features used by the SVM must be recomputed. Reusing source features from an old map would be incorrect.

For fidelity kernels, a symmetric N-point training Gram requires N(N-1)/2 pair circuits when the diagonal is known. M test points need MN additional pairs. Each pair uses compute-uncompute, approximately two state preparations before cancellation/compilation. Source replay bounds optimization cost but does not eliminate the final source-plus-target Gram cost.

For example, at IV-2a k=32, with 144 source examples, 32 source replay examples, a 112-trial suffix, 15 target SPSA steps, and 512 shots:

| Cost | Projected | Fidelity |
|---|---:|---:|
| Optimization objective evaluations | 46 | 46 |
| Optimization shots, replay+prefix size 64 | 4,521,984 | 47,480,832 |
| Final training construction shots, size 176 | 270,336 | 7,884,800 |
| Suffix prediction shots | 172,032 | 10,092,544 |
| Total target-stage shots in this example | 4,964,352 | 65,458,176 |

These are analytic counts for the specified scheme, not measured device throughput or resource estimates for a hidden faster quantum primitive. They exclude exact source initialization, calibration of the device, error mitigation overhead, retries and queue time. They illustrate why the projected model is the primary architecture even when fidelity is a useful scientific comparison.

State preparation is explicit angle encoding; no QRAM or free amplitude loading is assumed. q=4..8 exact simulation costs O(N L q 2^q), which is small. Equal trainable parameter counts therefore do not imply equal computational burden, and these experiments cannot establish a speed advantage over classical simulation.

## Hardware escalation

Only after a reproducible predictive/label-efficiency signal, choose representative cases by a rule fixed on development data: for example, predefined low/median/high classical session-shift strata, with all selected cases reported. Avoid selecting only quantum winners. Record device/backend ID, date, calibration, native basis, connectivity, transpiler seed, physical qubit mapping, compiled depth, two-qubit count, shots, mitigation, failures and end-to-end latency. Compare exact, finite-shot noiseless, device-noise simulation and hardware under the same frozen parameters before attempting noisy target optimization. Hardware feasibility is a separate result from statistical confirmation.
