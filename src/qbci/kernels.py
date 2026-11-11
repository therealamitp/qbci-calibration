"""PSD kernels, train-only spectral repair, and centered weighted alignment."""
import numpy as np
from scipy.spatial.distance import cdist


def rbf(x, y=None, gamma=1.0):
    return np.exp(-gamma * cdist(x, x if y is None else y, metric="sqeuclidean"))


def median_gamma(x):
    d = cdist(x, x, metric="sqeuclidean")
    values = d[np.triu_indices(len(x), 1)]
    positive = values[values > 1e-12]
    return 1.0 / np.median(positive) if len(positive) else 1.0


def weighted_center(k, weights):
    w = np.asarray(weights, float)
    if len(w) != len(k) or np.any(w < 0) or not w.sum() > 0:
        raise ValueError("Invalid centering weights")
    w = w / w.sum()
    return k - (k @ w)[:, None] - (w @ k)[None, :] + w @ k @ w


def alignment(k, y, weights=None):
    y = np.asarray(y)
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("Labels must be 0/1")
    if len(np.unique(y)) != 2:
        return 0.0
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    t = np.outer(2*y-1, 2*y-1)
    kc, tc = weighted_center(k, w), weighted_center(t, w)
    ww = w[:, None] * w[None, :]
    denominator = np.sqrt(np.sum(ww*kc*kc)*np.sum(ww*tc*tc))
    return float(np.sum(ww*kc*tc) / max(denominator, 1e-15))


class PSDRepair:
    """Clip a training Gram only, with a consistent out-of-sample extension.

    Never fit on a concatenated train+test matrix. Cross kernels are projected
    onto the retained training eigenspace, equivalent to explicit Nyström
    features K_test,train V Lambda^(-1/2).
    """
    def fit_transform(self, k):
        k = (k + k.T) / 2
        values, vectors = np.linalg.eigh(k)
        self.minimum_eigenvalue_ = float(values.min())
        keep = values > 1e-9
        self.vectors_ = vectors[:, keep]
        self.values_ = values[keep]
        self.negative_mass_ = float(-values[values < 0].sum())
        return (self.vectors_ * self.values_) @ self.vectors_.T

    def transform(self, cross):
        return (cross @ self.vectors_) @ self.vectors_.T


def diagnostics(k, y=None):
    eigenvalues = np.linalg.eigvalsh((k+k.T)/2)
    pos = np.clip(eigenvalues, 0, None)
    p = pos / max(pos.sum(), 1e-15)
    nonzero = p > 1e-15
    off = k[np.triu_indices(len(k), 1)]
    result = {"minimum_eigenvalue": float(eigenvalues.min()),
              "effective_rank": float(np.exp(-np.sum(p[nonzero]*np.log(p[nonzero])))),
              "participation_ratio": float(pos.sum()**2 / max(np.sum(pos**2), 1e-15)),
              "off_diagonal_mean": float(off.mean()) if len(off) else 0.0,
              "off_diagonal_variance": float(off.var()) if len(off) else 0.0}
    if y is not None:
        result["alignment"] = alignment(k, y)
    return result
