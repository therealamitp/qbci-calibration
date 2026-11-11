"""Bounded source-anchored kernel adaptation with explicit resource counts."""
from dataclasses import dataclass, asdict
import time
import numpy as np
from sklearn.svm import SVC
from . import quantum
from .kernels import rbf, median_gamma, alignment, PSDRepair, diagnostics
from .protocol import source_replay_indices, domain_weights, representation_update_allowed


@dataclass
class Resources:
    objective_calls: int = 0
    statevectors: int = 0
    circuit_settings: int = 0
    shots: int = 0
    one_qubit_gates: int = 0
    measurement_basis_gates: int = 0
    two_qubit_gates: int = 0
    gram_entries: int = 0
    seconds: float = 0.0


def spsa(objective, initial, active, bounds, steps=25, seed=0, learning_rate=0.08):
    """Exactly 3*steps+1 objective calls, same budget for every representation.

    Objective values select the best TRAINING iterate only. Validation/evaluation
    labels cannot enter this function. SPSA is an initial optimizer, not a claim
    that it is the optimal optimizer for every model.
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(initial, float).copy()
    active = np.asarray(active, int)
    bounds = np.asarray(bounds, float)
    value = objective(p)
    best, best_value, history = p.copy(), value, [float(value)]
    if not len(active):
        return best, history
    for i in range(steps):
        delta = rng.choice([-1.0, 1.0], size=len(active))
        c = 0.08 / (i+1)**0.101
        a = learning_rate / (i+1)**0.602
        plus, minus = p.copy(), p.copy()
        plus[active] += c*delta
        minus[active] -= c*delta
        plus = np.clip(plus, bounds[:, 0], bounds[:, 1])
        minus = np.clip(minus, bounds[:, 0], bounds[:, 1])
        gradient = (objective(plus)-objective(minus)) / (2*c) * delta
        p[active] -= a*np.clip(gradient, -10, 10)
        p = np.clip(p, bounds[:, 0], bounds[:, 1])
        value = objective(p)
        history.append(float(value))
        if value < best_value:
            best, best_value = p.copy(), value
    return best, history


def fourier_features(x, log_scales, harmonics=4):
    """Classical encoding control: harmonics plus all first-order pair products.

    The feature dictionary is prespecified. Paired sin/cos features make offsets
    cancel in Euclidean distance, so only scales are trainable here.
    """
    t = x * np.exp(log_scales)
    univariate = np.concatenate([f(k*t) for k in range(1, harmonics+1)
                                 for f in (np.sin, np.cos)], axis=1)
    pairs = []
    for j in range(x.shape[1]):
        for k in range(j+1, x.shape[1]):
            for f in (np.sin, np.cos):
                for g in (np.sin, np.cos):
                    pairs.append(f(t[:, j])*g(t[:, k]))
    result = np.column_stack([univariate] + pairs)
    return result / np.sqrt(result.shape[1])


class KernelAdapter:
    """Source model then independent budget-specific recalibration.

    kind: projected, fidelity, product (classical no-entangler twin), rbf,
    or fourier. `mode=classifier` freezes representation and refits only SVM;
    `subset` adapts encoding scales/offsets; `full` adapts all parameters.
    `shots` applies at adaptation AND final kernel construction. Source training
    stays exact and is explicitly identified as such in resource output.
    """
    def __init__(self, kind="projected", spec=None, C=1.0, gamma_multiplier=1.0,
                 source_steps=25, target_steps=15, anchor=0.1, replay=32,
                 target_mass=0.5, seed=0, shots=None, random_features=False):
        if kind not in ("projected", "fidelity", "product", "rbf", "fourier"):
            raise ValueError(kind)
        self.kind = kind
        self.spec = spec or quantum.CircuitSpec()
        if kind == "product" and self.spec.entangle:
            raise ValueError("Use an unentangled CircuitSpec for product")
        self.C, self.gamma_multiplier = C, gamma_multiplier
        self.source_steps, self.target_steps = source_steps, target_steps
        self.anchor, self.replay = anchor, replay
        self.target_mass, self.seed, self.shots = target_mass, seed, shots
        self.random_features = random_features
        self.resources = Resources()
        self.rng = np.random.default_rng(seed)

    def _initial(self):
        return (np.zeros(self.spec.input_dim) if self.kind in ("rbf", "fourier")
                else quantum.initial_parameters(self.spec, self.seed))

    def _bounds(self):
        return ([(-2.3, 1.1)]*self.spec.input_dim if self.kind in ("rbf", "fourier")
                else quantum.parameter_bounds(self.spec))

    def _representation(self, x, p, shots=None):
        if self.kind == "rbf":
            return x * np.exp(p)
        if self.kind == "fourier":
            repetitions = int(np.ceil(self.spec.qubits/self.spec.input_dim))
            return fourier_features(x, p, harmonics=2*self.spec.blocks*repetitions)
        if self.kind == "product":
            return quantum.product_expectations(x, p, self.spec)
        v = quantum.states(x, p, self.spec)
        self.resources.statevectors += len(x)
        if self.kind == "fidelity":
            return v
        settings = 3*len(x)
        repetitions = 0 if shots is None else settings*shots
        self.resources.circuit_settings += settings
        self.resources.shots += repetitions
        self.resources.one_qubit_gates += repetitions*self.spec.gates["one_qubit"]
        # X uses H; Y uses S-dagger then H; Z needs no basis rotation.
        self.resources.measurement_basis_gates += 0 if shots is None else 3*len(x)*shots*self.spec.qubits
        self.resources.two_qubit_gates += repetitions*self.spec.gates["two_qubit"]
        return (quantum.local_expectations(v, self.spec.qubits) if shots is None else
                quantum.sampled_expectations(v, self.spec.qubits, shots, self.rng))

    def _kernel(self, features, other=None, shots=None):
        n, m = len(features), len(features) if other is None else len(other)
        self.resources.gram_entries += n*m
        if self.kind != "fidelity":
            return rbf(features, other, self.gamma_)
        pairs = n*(n-1)//2 if other is None else n*m
        self.resources.circuit_settings += pairs
        repetitions = 0 if shots is None else pairs*shots
        self.resources.shots += repetitions
        # Compute-uncompute uses two state preparation circuits. Fixed final
        # entanglers can be compiled away; these are conservative logical counts.
        self.resources.one_qubit_gates += 2*repetitions*self.spec.gates["one_qubit"]
        self.resources.two_qubit_gates += 2*repetitions*self.spec.gates["two_qubit"]
        return (quantum.fidelity(features, other) if shots is None else
                quantum.sampled_fidelity(features, shots, self.rng, other))

    def fit_source(self, x, y):
        start = time.perf_counter()
        self.x_source_, self.y_source_ = np.asarray(x).copy(), np.asarray(y).copy()
        self.replay_indices_ = source_replay_indices(y, self.replay, self.seed)
        p = self._initial()
        f = self._representation(x, p)
        self.gamma_ = self.gamma_multiplier * (1.0 if self.kind == "fidelity" else median_gamma(f))
        replay_x, replay_y = x[self.replay_indices_], y[self.replay_indices_]
        def objective(params):
            self.resources.objective_calls += 1
            return -alignment(self._kernel(self._representation(replay_x, params)), replay_y)
        if self.random_features or self.source_steps == 0:
            self.source_parameters_, self.source_history_ = p.copy(), []
        else:
            self.source_parameters_, self.source_history_ = spsa(
                objective, p, np.arange(len(p)), self._bounds(), self.source_steps, self.seed)
        self.parameters_ = self.source_parameters_.copy()
        self._fit_classifier(x, y, domain_weights(len(x), 0), shots=None)
        self.resources.seconds += time.perf_counter()-start
        self.source_resources_ = asdict(self.resources)
        return self

    def _fit_classifier(self, x, y, weights, shots=None):
        self.train_features_ = self._representation(x, self.parameters_, shots)
        gram = self._kernel(self.train_features_, shots=shots)
        self.training_diagnostics_ = diagnostics(gram, y)
        self.repair_ = None
        if self.kind == "fidelity" and shots is not None:
            self.repair_ = PSDRepair()
            gram = self.repair_.fit_transform(gram)
        self.classifier_ = SVC(kernel="precomputed", C=self.C)
        # Keep total sample-weight mass equal to the fixed source count.
        self.classifier_.fit(gram, y, sample_weight=weights*len(self.x_source_))

    def adapt(self, x_target, y_target, mode="subset"):
        if mode not in ("frozen", "classifier", "subset", "full"):
            raise ValueError(mode)
        self.resources = Resources()
        self.rng = np.random.default_rng(self.seed + (0 if mode == "frozen" else 1009*len(y_target)))
        start = time.perf_counter()
        self.parameters_ = self.source_parameters_.copy()
        self.target_history_ = []
        self.update_reason_ = "classifier_only"
        if mode == "frozen":
            x_target, y_target = x_target[:0], y_target[:0]
            self.update_reason_ = "frozen_source"
        elif mode in ("subset", "full"):
            if representation_update_allowed(y_target):
                ids = self.replay_indices_
                x = np.concatenate([self.x_source_[ids], x_target])
                y = np.concatenate([self.y_source_[ids], y_target])
                weights = domain_weights(len(ids), len(y_target), self.target_mass)
                active = np.arange(len(self.parameters_) if mode == "full" else
                                   min(len(self.parameters_), 2*self.spec.input_dim))
                def objective(p):
                    self.resources.objective_calls += 1
                    k = self._kernel(self._representation(x, p, self.shots), shots=self.shots)
                    penalty = np.mean((p[active]-self.source_parameters_[active])**2)
                    return -alignment(k, y, weights) + self.anchor*penalty
                self.parameters_, self.target_history_ = spsa(
                    objective, self.parameters_, active, self._bounds(), self.target_steps,
                    self.seed + len(y_target))
                self.update_reason_ = "representation_adapted"
            else:
                self.update_reason_ = "insufficient_class_support"
        x = np.concatenate([self.x_source_, x_target])
        y = np.concatenate([self.y_source_, y_target])
        self._fit_classifier(x, y, domain_weights(len(self.x_source_), len(y_target), self.target_mass), self.shots)
        self.resources.seconds += time.perf_counter()-start
        return self

    def predict(self, x):
        start = time.perf_counter()
        f = self._representation(x, self.parameters_, self.shots)
        k = self._kernel(f, self.train_features_, self.shots)
        if self.repair_ is not None:
            k = self.repair_.transform(k)
        prediction = self.classifier_.predict(k)
        self.resources.seconds += time.perf_counter()-start
        return prediction
