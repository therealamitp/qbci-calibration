# Research program: can quantum feature adaptation reduce recalibration?

## Assessment

This is a worthwhile falsifiable research program, but the useful contribution is narrower than generic quantum EEG classification. Chronological cross-session QEEGNet results already exist, and quantaEEG already investigates session transfer and small-sample classification. The unresolved combination is a fixed chronological target-label budget, source-initialized trainable quantum representations, mechanistic classical controls, and independent confirmation. See the [prior-art table](prior_art.md) for evidence status and unresolved access limitations.

The appropriate initial conclusion is **unknown**, with a strong classical null hypothesis. At four to eight qubits, the proposed circuits are readily simulated. Their nonlinear encoding, source regularization or covariance alignment may help without any quantum-specific mechanism. A careful negative result that identifies the useful classical mechanism is preferable to an accuracy claim against an under-tuned baseline.

## Hypotheses and estimands

H1: the source-initialized projected-kernel model improves participant-average calibration AUC relative to the strongest relevant classical adaptation method chosen on development participants.

H2: the gain survives comparison with an identical-input adaptive RBF, encoding-matched trigonometric features, the no-entangler circuit, and a full-feature classical pipeline. H1 without H2 supports an engineering improvement but does not isolate a quantum representation benefit.

H3: benefits generalize under locked rules to OpenBMI and remain useful at a feasible shot and latency budget. This is a staged empirical hypothesis, not a presumption of computational speedup.

For participant i and target session s, let B_is(k) be balanced accuracy on the common suffix after the first 32 binary trials. The primary endpoint is

```
A_is = (1/32) sum_j (k_(j+1)-k_j) [B_is(k_j)+B_is(k_(j+1))]/2,
k = (0,4,8,16,32).
```

It is in balanced-accuracy units, not trial-accuracy units, with weights (2,4,6,12,8)/32 at the five budgets. A difference of 0.03 means three percentage points of normalized curve area. Average seeds within transition and transitions within participant; compare participants using paired differences. Sessions and seeds do not increase the independent sample size. Analyze each dataset separately; report a pooled development estimate only if participant independence is established and dataset weighting was specified in advance.

Because this linear-budget AUC weights later budgets more heavily than a log-budget plot, also report the entire curve, the 0..8 restricted AUC, and all budget-specific paired effects as secondary outcomes. A gain confined to 32 labels should not be described as strong evidence about four-label restoration.

## Prespecified secondary outcomes

Use balanced accuracy 0.75 as the proposed restoration target. It is a design benchmark, not a clinically established universal control threshold. Define the required label count as the first measured budget reaching the target with every larger measured budget also at or above it. Report the simpler first-crossing definition as a sensitivity analysis. Do not interpolate between tested budgets or smooth the curve to create a crossing.

Never encode a non-attainer as 32 successful labels. Report failures as greater than 32, the attainment probability at every budget, and the paired difference in attainment by 32. Describe label savings among joint attainers with the selection limitation stated. Include the full paired table: both attain, quantum only, classical only, neither. With only five sampled budgets, required labels are interval-bounded rather than known precisely.

Additional outcomes: per-budget balanced accuracy, class-specific recall, source-to-target degradation, recovery relative to the frozen source model, negative transfer, prediction/update latency, shots and executed circuits. A fully labeled target model can be an explicitly separate retrospective reference after analysis lock, never an oracle used to select the primary model or performance target.

Offline class decoding is a proxy for restored BCI control. It does not measure user learning, online closed-loop accuracy, throughput, fatigue or task completion time. Those require a later prospective online study.

## Statistical analysis

For each prespecified contrast, publish every participant's paired AUC effect, mean, median, win fraction and 95% interval. Use a paired t interval for the primary mean and a 10,000-resample participant bootstrap as a robustness summary; report discrepancies. With nine participants, bootstrap intervals can be unstable. Show the complete data, leave-one-participant-out sensitivity and session variation rather than presenting a narrow trial-level confidence interval.

OpenBMI supplies up to 54 independent participants, subject to data availability established before outcome analysis. The primary contrast is the projected subset model against the development-selected classical champion. Treat quantum-versus-other-classical superiority claims as a prespecified family and adjust p-values with Holm. Fidelity, full circuit adaptation, q/L sweeps, hardware subsets and mechanism associations are secondary or exploratory unless separately registered. No choosing the comparator after seeing OpenBMI results.

A conventional paired test of a zero mean difference does not establish the practically meaningful threshold. Report the interval relative to both zero and 0.03. Do not turn failure to reject a difference into an equivalence claim. Any equivalence/noninferiority analysis requires its own margin and test specification before confirmation.

Technical repeats use three locked seeds, averaged before participant inference. The prototype default is seed 0; the study driver must explicitly execute the registered seed list. Report optimization failures and numerical repair. For a technical failure, retry once under the identical recorded configuration if the cause is transient. Persistent methodological failures stop that study stage for diagnosis. Do not remove only unfavorable people or failed methods from the paired analysis. Before confirmation, lock either a universal failure policy with a frozen-source fallback or a declared non-estimable study outcome. The present runner fails loudly rather than silently applying a fallback.

### Power and precision

Use `scripts/power_sensitivity.py` for planning scenarios. With 54 people and a paired-effect standard deviation of 0.08, a true difference of 0.03 has roughly 79% normal-approximation power at two-sided alpha 0.05. If the standard deviation is 0.10, power is roughly 60%. These are assumptions, not estimated EEG power; correction for multiple comparisons and exclusions can reduce power. Nine-person cohorts are development and replication checks, not a persuasive sole basis for a small advantage. Recompute precision planning from development participant effects without inspecting OpenBMI outcomes.

## Mechanism tests

| Question | Planned diagnostic or ablation | Interpretation |
|---|---|---|
| Is the SVM refit sufficient? | Frozen vs classifier-only vs subset adaptation, same inputs and weights | An equal classifier-only result removes the representation-adaptation claim |
| Does alignment explain it? | Prefix alignment off/on for quantum and matched classical maps, full-feature RA baseline | Disappearance after shared alignment supports a classical shift-correction mechanism |
| Does compression explain it? | Full tangent RBF vs four-dimensional RBF and quantum; source PCA retained variance and target clipping rate | A quantum win only against compressed baselines may not improve practical BCI |
| Does simple metric learning explain it? | Adaptive diagonal-scale RBF with matched source anchor and optimization budget | Matching performance supports a classical rescaling explanation |
| Does the encoding explain it? | Trigonometric feature map, exact no-entangler product-state twin | Matching performance supports a classical nonlinear representation mechanism |
| Is entanglement useful? | Same circuit with CNOTs removed, common input, seed and update policy | A gain requires an ablation difference, not merely the presence of gates |
| Are learned maps better than fixed ones? | Random fixed map, source-trained frozen map, target-adapted map | Separates initialization, source learning and target learning |
| What changed geometrically? | Held-out class-centroid distance/within-class scatter, KTA, kernel spectrum/effective rank, off-diagonal variance and source-target MMD | Descriptive evidence for separation or shift sensitivity; not proof of causal mechanism |
| Does noise erase the effect? | Exact vs 128/512/2048 shots at both target optimization and deployment; frozen-theta sampling ablation | Separates optimization noise from measurement noise |

Compute any supervised suffix diagnostic only after predictions and choices are locked. Do not feed test KTA back to the optimizer. Unsupervised suffix MMD is also a retrospective analysis in the primary regime, not permission to normalize on the suffix. Compare changes against controls and participant effects, not just attractive embeddings. The supplied `mechanisms.py` provides reusable metrics; a complete mechanism-report generator is still an extension.

## Prioritized experiment matrix

| Priority | Dataset / scale | Models and intervention | Purpose / exit condition |
|---|---|---|---|
| 0 | Synthetic data and numerical probes | Circuit sensitivity/invariance, all default baselines, one-class prefixes, suffix isolation, finite shots | All integrity tests pass before real-data work. This stage produces no EEG evidence |
| 1 | IV-2a, nine people | 4q/2-block projected subset, frozen/classifier/full; compressed RBF, adaptive RBF, full tangent RA, CSP/FBCSP, EEGNet | Establish data audit and whether there is any credible calibration signal under subject-disjoint tuning |
| 2 | IV-2a and IV-2b | Fidelity comparison, fixed maps, trigonometric map, no-entangler and classical twin; shared-prefix alignment factorial; complete RPA/TSA implementations | Isolate the representation and compare against mature adaptation methods. Stop an advantage narrative if the classical controls explain it |
| 3 | IV-2b, all five sessions | Fixed session-1 source to later sessions; separate feedback strata and secondary adjacent-session design | Test repeatability and three-channel robustness; do not multiply n by number of sessions |
| 4 | Development only | q=6/8, L=1/3 sensitivities; exact vs 128/512/2048 shots; constrained optimizer-budget comparisons | Choose a bounded architecture and resource envelope before OpenBMI |
| 5 | Locked OpenBMI, up to 54 people | Prespecified primary projected map, classical champion and mechanism controls; 0/4/8/16/32 | Independent confirmation under an archived protocol, with all failures and uncertainty |
| 6 | Prespecified representative cases | Actual hardware, calibration metadata, transpilation, identical shots; no favorable-case selection | Feasibility only unless a suitably powered hardware evaluation is performed |
| 7 | New study | Prospective online control, cross-subject initialization or drift-triggered labels | Separate scientific questions and budgets |

## Tuning fairness

Start with at most 24 development configurations per model family, selected before evaluating outer folds. Give classical SVM/RA methods their own relevant C, bandwidth, alignment shrinkage and target weighting choices. Give CSP/LDA shrinkage and component/filter choices; EEGNet learning-rate, training-epoch and constrained fine-tuning choices. Hold candidate count and validation access comparable while reporting total compute. A fixed 24-point Cartesian quantum search compared with one classical default is not a fair test.

The current configuration object exposes the shared kernel choices but not every baseline-specific tuning parameter. Extending it is a required implementation step before claiming the strongest classical benchmark. The code is an executable starting point, not evidence that those tuning obligations have already been fulfilled.

## Continuation and reporting rules

Proposed minimum practically meaningful effect: **0.03 absolute normalized AUC**, chosen before confirmation. Continue toward a full advantage-focused study only when development results survive mature classical comparisons and the independent OpenBMI result has an estimated gain at least 0.03 with a 95% interval excluding zero. Require a positive participant median and gains in at least 60% of confirmation participants as prespecified heterogeneity checks. Report whether the interval's lower bound exceeds 0.03; if it does not, the evidence does not establish that the true gain clears the practical threshold.

Require the result to survive the practical full-feature comparator and matched encoding/adaptation controls. If another prespecified classical method erases the gain, narrow the claim even if the development-selected primary contrast is significant. A label-efficiency claim additionally requires favorable target-attainment outcomes, with non-attainers included; a suggested useful secondary change is at least one eight-trial step among joint attainers, without a worse attainment rate. This is not a substitute for the primary endpoint.

If the projected model fails these gates, stop escalation to costly hardware or broad circuit searches. Publish the tested range, intervals and useful classical mechanism. A noisy or inconclusive result is neither proof of advantage nor proof of universal impossibility.

Distinguish four claims throughout:

1. **Predictive improvement:** higher performance under the same recorded data access.
2. **Label efficiency:** better calibration curves and attainment with fewer revealed labels.
3. **Hardware feasibility:** executable circuits with tolerable shots, latency and device noise.
4. **Computational quantum advantage:** a defensible computational benefit over strong classical algorithms, including simulation. The proposed small circuits do not establish this claim.

## Staged implementation plan

The repository implements the numerical and protocol core now. Next, audit and freeze one complete raw-data export for each regime, extend baseline-specific tuning and published RPA/TSA, run development folds, and record the chosen settings. Then run shot/resource studies, archive a protocol and environment lock, and execute OpenBMI once. Hardware and online control follow only if the corresponding evidence justifies them. See [implementation coverage](implementation.md) for the exact distinction between delivered code, executed tests and unexecuted scientific experiments.
