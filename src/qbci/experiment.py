"""Primary fixed-suffix experiments. No target suffix enters any fit method."""
from dataclasses import dataclass, asdict
import time
import numpy as np
from .quantum import CircuitSpec
from .adaptation import KernelAdapter
from .classical import FBCSP, TangentProcrustes, fit_replay_classifier
from .preprocessing import covariance_epochs, SourceFeatures
from .kernels import median_gamma
from .protocol import prefix_plans, validate_pair
from .statistics import balanced_accuracy


@dataclass
class ExperimentConfig:
    source_steps: int = 25
    target_steps: int = 15
    C: float = 1.0
    gamma_multiplier: float = 1.0
    anchor: float = 0.1
    target_mass: float = 0.5
    replay: int = 32
    dimensions: int = 4
    qubits: int = 4
    blocks: int = 2
    seed: int = 0
    shots: object = None
    source_epochs: int = 100
    target_epochs: int = 20


DEFAULT_METHODS = (
    "projected_frozen", "projected_classifier", "projected_subset", "projected_full",
    "projected_random", "unentangled_subset", "product_subset",
    "fidelity_frozen", "fidelity_classifier", "fidelity_subset",
    "rbf_frozen", "rbf_classifier", "rbf_subset", "fourier_subset",
    "tangent_frozen", "tangent_classifier", "riemann_alignment", "tangent_procrustes",
    "csp_lda", "fbcsp_lda")


def run_pair(source, target, config=None, methods=DEFAULT_METHODS):
    config = config or ExperimentConfig()
    validate_pair(source, target)
    plans = prefix_plans(len(target.labels))
    # Per-trial operations are stateless, so it is safe to cache covariances.
    # No aggregate of target data is used here or by SourceFeatures.fit.
    t0 = time.perf_counter()
    cs = covariance_epochs(source.epochs, source.sfreq)
    ct = covariance_epochs(target.epochs, target.sfreq)
    basis = SourceFeatures(config.dimensions).fit(cs)
    xs, xt = basis.transform(cs), basis.transform(ct)
    fs, ft = basis.transform_full(cs), basis.transform_full(ct)
    preprocessing_seconds = time.perf_counter()-t0
    source_models, rows, predictions, resources = {}, [], [], []
    for method in methods:
        source_start = time.perf_counter()
        source_cache_hit = False
        if method.startswith("eegnet_"):
            from .eegnet import EEGNetAdapter
            model = EEGNetAdapter(seed=config.seed, source_epochs=config.source_epochs,
                                  target_epochs=config.target_epochs, target_mass=config.target_mass)
            model.fit_source(source.epochs, source.labels, source.sfreq)
            category = "eegnet"
        elif method in ("csp_lda", "fbcsp_lda"):
            model = FBCSP(bands=((8, 30),)) if method == "csp_lda" else FBCSP()
            model.fit(source.epochs, source.labels, source.sfreq)
            csp_source, csp_target = model.transform(source.epochs), model.transform(target.epochs)
            category = "csp"
        elif method.startswith("tangent_") or method == "riemann_alignment":
            category = "tangent"
            model = TangentProcrustes().fit_source(fs, source.labels) if method == "tangent_procrustes" else None
            gamma_full = config.gamma_multiplier*median_gamma(fs)
        else:
            category = "kernel"
            kind, mode = method.rsplit("_", 1)
            entangle = kind not in ("unentangled", "product")
            kind = "projected" if kind == "unentangled" else kind
            key = (kind, entangle, mode == "random")
            if key not in source_models:
                model = KernelAdapter(kind=kind,
                    spec=CircuitSpec(config.qubits, config.blocks, config.dimensions, entangle),
                    C=config.C, gamma_multiplier=config.gamma_multiplier,
                    source_steps=config.source_steps, target_steps=config.target_steps,
                    anchor=config.anchor, replay=config.replay, target_mass=config.target_mass,
                    seed=config.seed, shots=config.shots, random_features=mode == "random")
                model.fit_source(xs, source.labels)
                source_models[key] = model
            else:
                source_cache_hit = True
            model = source_models[key]
        source_setup_seconds = time.perf_counter()-source_start
        for plan in plans:
            k, indices, test = plan.budget, plan.calibration, plan.evaluation
            labels = target.labels[indices]
            start = time.perf_counter()
            details = {}
            if category == "kernel":
                mode = method.rsplit("_", 1)[1]
                model.adapt(xt[indices], labels, "classifier" if mode == "random" else mode)
                pred = model.predict(xt[test])
                details = {"source": model.source_resources_, "target": asdict(model.resources),
                           "reason": model.update_reason_, "diagnostics": model.training_diagnostics_,
                           "source_history": model.source_history_, "target_history": model.target_history_}
            elif category == "csp":
                clf = fit_replay_classifier(csp_source, source.labels, csp_target[indices], labels,
                                            kind="lda", target_mass=config.target_mass)
                pred = clf.predict(csp_target[test])
            elif category == "eegnet":
                model.adapt(target.epochs[indices], labels, method.split("_")[1])
                pred = model.predict(target.epochs[test])
            else:
                train, calibration, evaluation, gamma = fs, ft[indices], ft[test], gamma_full
                if method == "tangent_frozen":
                    calibration, labels = calibration[:0], labels[:0]
                elif method in ("riemann_alignment", "tangent_procrustes"):
                    refs = basis.prefix_reference(ct[indices])
                    calibration = basis.transform_full(ct[indices], refs) if k else ft[indices]
                    evaluation = basis.transform_full(ct[test], refs)
                    if method == "tangent_procrustes":
                        model.adapt(calibration, labels)
                        train, calibration, evaluation = model.source_, model.transform(calibration), model.transform(evaluation)
                        gamma = config.gamma_multiplier*median_gamma(train)
                clf = fit_replay_classifier(train, source.labels, calibration, labels, C=config.C,
                                            gamma=gamma, target_mass=config.target_mass)
                pred = clf.predict(evaluation)
            row = {"dataset": source.dataset, "subject": source.subject,
                   "source_session": source.session, "target_session": target.session,
                   "source_feedback": source.feedback, "target_feedback": target.feedback,
                   "method": method, "budget": k, "seed": config.seed,
                   "balanced_accuracy": balanced_accuracy(target.labels[test], pred),
                   "n_evaluation": len(test), "seconds": time.perf_counter()-start,
                   "source_preprocessing_seconds": preprocessing_seconds,
                   "source_setup_seconds": source_setup_seconds,
                   "source_cache_hit": source_cache_hit,
                   "target_left": int(np.count_nonzero(target.labels[indices] == 0)),
                   "target_right": int(np.count_nonzero(target.labels[indices] == 1))}
            rows.append(row)
            for i, prediction in zip(test, pred):
                predictions.append({**{name: row[name] for name in ("dataset", "subject", "source_session",
                    "target_session", "method", "budget", "seed")}, "trial_id": str(target.trial_ids[i]),
                    "truth": int(target.labels[i]), "prediction": int(prediction)})
            resources.append({**row, **details})
    return rows, predictions, resources
