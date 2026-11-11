"""Small, auditable batched statevector circuits and joint Pauli sampling.

Wire 0 is the most significant computational-basis bit. Each block applies
RY(t_j), RZ(t_j), then even-edge/odd-edge nearest-neighbor CNOTs. Between
blocks apply RX, RY, RZ trainable rotations. There is NO terminal trainable
unitary. Encoding scales and offsets are shared across reuploading blocks.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CircuitSpec:
    qubits: int = 4
    blocks: int = 2
    input_dim: int = 4
    entangle: bool = True

    def __post_init__(self):
        if self.qubits < 1 or self.blocks < 1 or not 1 <= self.input_dim <= self.qubits:
            raise ValueError("Require qubits >= input_dim >= 1 and blocks >= 1")

    @property
    def n_parameters(self):
        return 2 * self.input_dim + 3 * self.qubits * (self.blocks - 1)

    @property
    def gates(self):
        return {"one_qubit": 2 * self.qubits * self.blocks +
                3 * self.qubits * (self.blocks - 1),
                "two_qubit": (self.qubits - 1) * self.blocks if self.entangle else 0}


def initial_parameters(spec, seed=0):
    rng = np.random.default_rng(seed)
    p = np.zeros(spec.n_parameters)
    p[2 * spec.input_dim:] = rng.normal(0, 0.15, len(p) - 2 * spec.input_dim)
    return p  # log scales = 0, offsets = 0


def parameter_bounds(spec):
    return ([(-2.3, 1.1)] * spec.input_dim +
            [(-np.pi, np.pi)] * (spec.n_parameters - spec.input_dim))


def rotation(axis, angles):
    a = np.atleast_1d(angles)
    c, s = np.cos(a / 2), np.sin(a / 2)
    g = np.zeros((len(a), 2, 2), dtype=complex)
    if axis == "x":
        g[:, 0, 0] = g[:, 1, 1] = c
        g[:, 0, 1] = g[:, 1, 0] = -1j * s
    elif axis == "y":
        g[:, 0, 0] = g[:, 1, 1] = c
        g[:, 0, 1], g[:, 1, 0] = -s, s
    elif axis == "z":
        g[:, 0, 0], g[:, 1, 1] = np.exp(-0.5j * a), np.exp(0.5j * a)
    else:
        raise ValueError(axis)
    return g


def apply_single(states, gate, wire, n):
    g = np.asarray(gate)
    if g.ndim == 2:
        g = g[None]
    v = states.reshape(len(states), 2 ** wire, 2, 2 ** (n - wire - 1))
    return np.einsum("nij,najk->naik", g, v).reshape(states.shape)


def apply_cnot(states, control, target, n):
    idx = np.arange(2 ** n)
    mask = (idx >> (n - 1 - control)) & 1
    return states[:, idx ^ (mask << (n - 1 - target))]


def states(x, parameters, spec):
    x, p = np.asarray(x, float), np.asarray(parameters, float)
    if x.ndim != 2 or x.shape[1] != spec.input_dim or p.shape != (spec.n_parameters,):
        raise ValueError("Feature or parameter shape does not match CircuitSpec")
    if not np.isfinite(x).all() or not np.isfinite(p).all():
        raise ValueError("Nonfinite circuit inputs")
    d, n = spec.input_dim, spec.qubits
    angles = x * np.exp(p[:d]) + p[d:2*d]
    v = np.zeros((len(x), 2 ** n), complex)
    v[:, 0] = 1
    for block in range(spec.blocks):
        for wire in range(n):
            for axis in ("y", "z"):
                v = apply_single(v, rotation(axis, angles[:, wire % d]), wire, n)
        if spec.entangle:
            for parity in (0, 1):
                for wire in range(parity, n - 1, 2):
                    v = apply_cnot(v, wire, wire + 1, n)
        if block < spec.blocks - 1:
            for wire in range(n):
                for axis_index, axis in enumerate(("x", "y", "z")):
                    value = p[2*d + 3*(block*n + wire) + axis_index]
                    v = apply_single(v, rotation(axis, value), wire, n)
    return v


def local_expectations(v, n):
    out = []
    for wire in range(n):
        a = v.reshape(len(v), 2 ** wire, 2, 2 ** (n-wire-1))
        cross = np.sum(a[:, :, 0].conj() * a[:, :, 1], axis=(1, 2))
        z = np.sum(abs(a[:, :, 0])**2 - abs(a[:, :, 1])**2, axis=(1, 2))
        out.extend((2*cross.real, 2*cross.imag, z))
    return np.stack(out, axis=1)


def sampled_expectations(v, n, shots, rng):
    """Three joint measurement settings; `shots` is PER setting per state.

    Sampling whole bitstrings preserves correlations between local observables
    in the same measurement basis. Independent Bernoulli draws would not.
    """
    if shots < 1:
        raise ValueError("shots must be positive")
    h = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    sd = np.diag([1, -1j])
    signs = 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n-1, -1, -1)) & 1)
    out = np.empty((len(v), n, 3))
    for j, basis in enumerate((h, h @ sd, np.eye(2))):
        w = v.copy()
        for wire in range(n):
            w = apply_single(w, basis, wire, n)
        for i, probabilities in enumerate(abs(w)**2):
            probabilities = probabilities / probabilities.sum()
            counts = rng.multinomial(shots, probabilities)
            out[i, :, j] = counts @ signs / shots
    return out.reshape(len(v), 3*n)


def product_expectations(x, p, spec):
    """Exact classical O(n*blocks) twin of the no-entanglement circuit."""
    if spec.entangle:
        raise ValueError("Product twin requires entangle=False")
    d, n = spec.input_dim, spec.qubits
    angles = x * np.exp(p[:d]) + p[d:2*d]
    outputs = []
    for wire in range(n):
        v = np.zeros((len(x), 2), complex)
        v[:, 0] = 1
        for block in range(spec.blocks):
            for axis in ("y", "z"):
                v = apply_single(v, rotation(axis, angles[:, wire % d]), 0, 1)
            if block < spec.blocks - 1:
                for a, axis in enumerate(("x", "y", "z")):
                    v = apply_single(v, rotation(axis, p[2*d+3*(block*n+wire)+a]), 0, 1)
        outputs.append(local_expectations(v, 1))
    return np.concatenate(outputs, axis=1)


def fidelity(v, w=None):
    w = v if w is None else w
    return np.clip(abs(v.conj() @ w.T)**2, 0, 1)


def sampled_fidelity(v, shots, rng, w=None):
    k = fidelity(v, w)
    if w is not None:
        return rng.binomial(shots, k) / shots
    i, j = np.triu_indices(len(v), 1)
    out = np.eye(len(v))
    out[i, j] = rng.binomial(shots, k[i, j]) / shots
    out[j, i] = out[i, j]
    return out
