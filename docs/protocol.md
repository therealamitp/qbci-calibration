# Dataset and leakage-resistant evaluation protocol

## Dataset roles and timing

| Dataset | Verified structure and chosen data | Primary transfer | Cue-relative analysis window | Expected binary trials and evaluation suffix |
|---|---|---|---|---|
| BCI Competition IV-2a, `BNCI2014_001` | Nine people, two recording days, 22 EEG channels, 250 Hz, four imagery classes. Retain left/right hand only, preserving original order | Session 1 to session 2 | Filter from cue onset; analyze [0.5, 3.5) seconds. The inspected loader's trial-start events require a +2 s cue offset | 144 left/right trials per session; 112 evaluation trials after the first 32 |
| BCI Competition IV-2b, `BNCI2014_004` | Nine people, five sessions, three bipolar EEG channels, 250 Hz. Sessions 1-2 screening without feedback, sessions 3-5 feedback | Fixed session-1 source to each of sessions 2-5 | Same [0.5, 3.5) window; resolve the native marker against the pinned loader and original paradigm before specifying offsets | 120 trials in each of sessions 1-2; 160 in each of sessions 3-5. Suffix 88 or 128 |
| OpenBMI, `Lee2019_MI` | 54 people, two sessions, 62 EEG channels, 1000 Hz; left/right imagery. Use `train_run=True`, `test_run=False`, `resting_state=False` | Session 1 training run to session 2 training run, after architecture/analysis lock | Same [0.5, 3.5) window. Inspected MI loader is cue-relative, interval [0,4], hence offset 0 | Expected 100 labeled training trials per session; suffix 68. Verify against downloaded release |

Sources: [MOABB IV-2a](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html), [MOABB IV-2b](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_004.html), [original IV-2b description](https://www.bbci.de/competition/iv/desc_2b.pdf), [MOABB OpenBMI](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Lee2019_MI.html), [OpenBMI dataset paper](https://doi.org/10.1038/s41597-019-0086-2).

Some generated MOABB IV-2b prose says two sessions despite the original five-session protocol. The raw loader and source documentation must govern the audit. Do not inherit a generated-doc error. Feedback status is associated with a changed task/display regime and acquisition settings, so a feedback transition is not a pure measurement of spontaneous neural drift.

For OpenBMI, the documentation warns that MI test runs do not have usable ground-truth labels. The inspected loader can read a `y_dec` field when constructing runs, but its presence alone is not evidence that online intended-class labels are available. Test-run classifier decisions must never be treated as ground truth. Training-only analysis is the conservative primary specification. Extending to online runs requires tracing the release fields to original intended cues and registering that change before evaluating models.

Inspected loader files: MOABB `moabb/datasets/Lee2019.py`, blob `81da8b41bf1b61d808c4507b80ad6f670e6ee648`, and `moabb/datasets/bnci/bnci_2014.py`, blob `80003cabc5acfe461f5dec7936f2e2ef0d67d0f8`. These identify the inspected online snapshot; they are not the installed version of an executed EEG experiment. Raw data have not been downloaded and audited in this release.

## Data audit before model fitting

1. Pin Python, MOABB, MNE, NumPy, SciPy, scikit-learn and any neural/quantum backend. Record the complete Git commit and environment lock.
2. Download original files under their access terms. Store local file checksums, release/URL, subject IDs, session order, run order, channel names/types, units, sample rates, event dictionaries and cue offsets. The current exporter hashes its exported sessions and loader class; extend the manifest with original-file checksums before a scientific run.
3. Check chronological event samples and native cue markers manually on at least one example per dataset and every distinct IV-2b session regime. Do not infer chronology from arbitrary lexical session names.
4. Verify counts and left/right mapping. OpenBMI uses a different native class-number convention; exported labels must consistently be left=0, right=1. Audit all expected counts without selecting favorable participants.
5. Exclude EOG/EMG/stim channels. Keep the full designated EEG montage within each dataset. Do not reduce 62-channel OpenBMI to a favorable subset based on target outcomes.
6. Retain all trials passing predeclared technical validity criteria. No performance-based participant exclusion. Do not remove low-accuracy people. Missing or corrupt epochs must trigger an explicit audit decision; never silently backfill the first 32 with later labeled observations.

The initial exporter deliberately fails on nonfinite/incomplete epochs and unexpected counts. After an audit, a study may define a quality policy before model evaluation, recording both elapsed cues and eligible trials. Rejection or abstention behavior must be identical across methods. The current protocol does not claim to have solved online artifact rejection.

## Shared representation

Each trial is exported from cue onset to four seconds. Apply independent causal fourth-order Butterworth SOS filters for 8-13 Hz and 13-30 Hz, resetting state at cue onset. Compute OAS shrinkage covariances on [0.5,3.5). No forward-backward filtering, across-session normalization, or target-dependent bad-channel selection is used.

For each band, estimate a source affine-invariant covariance mean. Whiten/log-map each covariance at that reference, vectorize its upper triangle with sqrt(2) weighting on off-diagonal entries, and concatenate bands. Fit source-only feature scaling, PCA and reduced-coordinate scaling. Freeze all of them in the principal comparison. Full tangent dimensions are 506, 12 and 3906 for IV-2a, IV-2b and OpenBMI respectively. The four-dimensional quantum input can lose useful information; compare it with the full-feature classical pipeline, not only with equally compressed models.

The primary frozen-basis quantum experiment has no target covariance alignment update. A separate classical baseline estimates a target reference using only the allowed prefix, geodesically shrunk toward the source mean with weight k/(k+8). Source scaling and feature coordinates remain fixed. Add a secondary factorial experiment giving quantum, RBF and trigonometric models the **same** prefix alignment, to test whether a gain persists after this classical mechanism. That factorial arm is planned, not yet exposed in the default runner.

All trial features used for a prediction are available by cue+3.5 seconds. Convolutional or covariance processing within that completed window does not imply continuous within-trial feedback. The code may compute target covariances in a batch for efficiency, but these operations are per-trial and stateless; no aggregate is fitted on future or skipped target trials. Cached future features must not enter adaptive normalization. A truly streaming system requires a separately specified decision latency and causal state management.

## Primary fixed-suffix experiment

Budgets are exactly k in {0,4,8,16,32}. For every target session:

| Data block | Permitted use at budget k |
|---|---|
| Entire allowed source session | Fit preprocessing, feature map and classifier; source replay |
| Target trials 1..k | Calibration features and labels; permitted target alignment only in the declared arm |
| Target trials k+1..32 | Unused. No fitting, normalization, pseudo-labeling, early stopping or rejection-driven replacement |
| Target trials 33..end | Common untouched evaluation suffix for all budgets; labels used only after predictions are fixed |

Use chronological prefixes, never a future-balanced selection. At k=0, no target-session statistics are available. Every budget restarts from the same source solution; it is an independent calibration experiment, not a warm-start continuation from k=4 to k=8. This makes the label-efficiency curve interpretable at a known budget.

A one-class prefix is not a reason to resample. Use source replay for the two-class SVM/LDA fit. Freeze representation updates until at least two target examples of each class have appeared. Record the observed class counts and the reason an update was skipped. Keep the actual labels in the classifier fit. This conservative support rule applies to quantum and matched trainable classical maps.

Keep the exact suffix identifiers equal across methods and budgets. A suffix lacking either class is a prespecified non-estimable balanced-accuracy case, reported rather than repaired by selecting different observations. The runner fails instead of silently substituting ordinary accuracy.

## Repeated sessions and feedback

The main IV-2b design uses the same fully labeled session 1 as source for targets 2,3,4,5. Average those four target-session endpoints within participant for the dataset-level summary. Report each session and feedback status separately. This measures repeated session-boundary restoration from a common baseline without acquiring labels from earlier evaluation suffixes.

A secondary adjacent-session analysis can use 1->2, 2->3, 3->4, 4->5, with each earlier session declared as a fully labeled historical source. It answers a different question and must disclose the extra historical labels. It is not a continuous stream with only 32 labels total. A strict rolling replay may carry forward only labels actually revealed in prior sessions, with a cumulative label ledger. Never turn previous evaluation labels into free training data.

In adjacent analyses, stratify no-feedback->no-feedback (1->2), no-feedback->feedback (2->3), and feedback->feedback (3->4,4->5). Do not treat those transitions as independent participants or interpret all changes as the same physiological drift process.

## Separate sequential replay

The replay analysis scores each target trial before revealing any requested label. Reveal the first k labels, count every reveal, and update after every four reveals or the final reveal. Prefix observations therefore contribute to measured performance under the model that was available at the time. Preserve an event log proving prediction precedes reveal. Report cumulative and blockwise balanced accuracy, label count and update latency separately from the primary fixed-suffix curve.

Session-boundary calibration is the primary intervention. Drift-triggered label requests are optional future work. Such a detector may use past input statistics, predictions or uncertainty, but not unrevealed outcomes. Tune its trigger and label schedule on development subjects; enforce the same maximum budget and request opportunities across methods. Charge detector computation and any queried label. Do not use oracle-selected drift times.

## Development and independent confirmation

Use subject-disjoint development evaluation on IV-2a and IV-2b. For each outer held-out participant, choose settings using only other development participants' sessions. The held-out person's source session is still allowed to fit their personalized model; their target suffix is never used to select settings. Session split alone is not a substitute for subject-disjoint hyperparameter selection.

Select the primary classical comparator using development participant-average calibration AUC from the complete relevant baseline set. Also retain all prespecified classical comparisons. Match the number of candidate configurations and record actual time and objective calls. Quantum circuits receive no extra search because their parameterization is unfamiliar. Search bounds and any failed candidates are logged.

After development, lock architecture, preprocessing, comparator, adaptation objective, seeds, regularization, optimization budget, statistical contrasts, stopping rules and failure handling. Archive a timestamped protocol externally as well as a hash. A local JSON hash cannot prove when a decision was made.

OpenBMI is an independent confirmation cohort. Fit new within-person source models from each participant's session 1 under the locked rules, then evaluate their session 2. This is **dataset replication**, not direct transfer of a trained IV-2a model to OpenBMI. Cross-subject initialization is a separate extension with its own disjoint donors and label accounting. Check whether source collections share participant identities before pooling any development subjects across datasets; do not assume that different numeric IDs prove independence.
