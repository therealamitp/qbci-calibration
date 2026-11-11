"""Causal epoch preprocessing and source-fitted covariance representations."""
import numpy as np
from scipy.signal import butter, sosfilt
from sklearn.covariance import oas
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def sym(a):
    return (a + np.swapaxes(a, -1, -2)) / 2


def spd_function(a, function):
    values, vectors = np.linalg.eigh(sym(a))
    floor = np.maximum(values.max(axis=-1, keepdims=True) * 1e-10, 1e-20)
    values = np.maximum(values, floor)
    return (vectors * function(values)[..., None, :]) @ np.swapaxes(vectors, -1, -2)


def matrix_exp(a):
    values, vectors = np.linalg.eigh(sym(a))
    return (vectors * np.exp(np.clip(values, -50, 50))[..., None, :]) @ np.swapaxes(vectors, -1, -2)


def riemann_mean(covs, iterations=50, tolerance=1e-8):
    """Affine-invariant Karcher mean with a damped log-map iteration."""
    mean = covs.mean(axis=0)
    for _ in range(iterations):
        root = spd_function(mean, np.sqrt)
        inverse = spd_function(mean, lambda x: 1/np.sqrt(x))
        delta = spd_function(inverse @ covs @ inverse, np.log).mean(axis=0)
        if np.linalg.norm(delta) < tolerance:
            break
        mean = sym(root @ matrix_exp(0.5 * delta) @ root)
    return mean


def geodesic(a, b, fraction):
    root = spd_function(a, np.sqrt)
    inv = spd_function(a, lambda x: 1/np.sqrt(x))
    return sym(root @ spd_function(inv @ b @ inv, lambda x: x ** fraction) @ root)


def tangent(covs, reference):
    inv = spd_function(reference, lambda x: 1/np.sqrt(x))
    logs = spd_function(inv @ covs @ inv, np.log)
    row, col = np.triu_indices(logs.shape[-1])
    scale = np.where(row == col, 1.0, np.sqrt(2))
    return logs[:, row, col] * scale


def causal_filter_epochs(epochs, sfreq, bands=((8, 13), (13, 30)), order=4):
    """Reset filter state for each trial to exclude intervening trial samples.

    Input begins at cue onset; discard the first 0.5 seconds downstream.
    No forward-backward filter, resampling, baseline fit or across-trial state.
    """
    x = np.asarray(epochs, float)
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError("epochs must be finite [trial, channel, time]")
    return np.stack([sosfilt(butter(order, band, btype="bandpass", fs=sfreq,
                                    output="sos"), x, axis=-1)
                     for band in bands], axis=1)


def covariance_epochs(epochs, sfreq, bands=((8, 13), (13, 30)), window=(0.5, 3.5)):
    start, end = (int(round(t * sfreq)) for t in window)
    if start < 0 or end > epochs.shape[-1] or end <= start:
        raise ValueError("Cue-relative window falls outside the available epoch")
    filtered = causal_filter_epochs(epochs, sfreq, bands)
    covs = np.empty(filtered.shape[:2] + (epochs.shape[1], epochs.shape[1]))
    for i in range(len(epochs)):
        for band in range(len(bands)):
            segment = filtered[i, band, :, start:end]
            covs[i, band] = oas(segment.T, assume_centered=False)[0]
    return covs


class SourceFeatures:
    """Source-fitted tangent basis, scaling, PCA and angle scaling.

    `fit` never takes target data. All updates are explicit in transform's
    optional reference argument. PCA/scalers remain frozen across budgets.
    """
    def __init__(self, dimensions=4):
        self.dimensions = dimensions

    def fit(self, covs):
        self.references_ = np.stack([riemann_mean(covs[:, b]) for b in range(covs.shape[1])])
        features = self.full(covs)
        self.scaler_ = StandardScaler().fit(features)
        z = self.scaler_.transform(features)
        if self.dimensions > min(z.shape):
            raise ValueError("Requested PCA dimension exceeds source rank bound")
        self.pca_ = PCA(n_components=self.dimensions, svd_solver="full").fit(z)
        reduced = self.pca_.transform(z)
        self.reduced_scaler_ = StandardScaler().fit(reduced)
        self.n_source_ = len(covs)
        return self

    def full(self, covs, references=None):
        refs = self.references_ if references is None else references
        return np.concatenate([tangent(covs[:, b], refs[b]) for b in range(covs.shape[1])], axis=1)

    def transform_full(self, covs, references=None):
        return self.scaler_.transform(self.full(covs, references))

    def transform(self, covs, references=None):
        z = self.pca_.transform(self.transform_full(covs, references))
        # Fixed, source-fitted clipping and angular conversion. Count clipping
        # separately in an audit; never re-estimate from the target suffix.
        return np.clip(self.reduced_scaler_.transform(z), -3, 3) * (np.pi / 3)

    def prefix_reference(self, prefix_covs, prior_trials=8.0):
        if len(prefix_covs) == 0:
            return self.references_.copy()
        alpha = len(prefix_covs) / (len(prefix_covs) + prior_trials)
        return np.stack([geodesic(self.references_[b], riemann_mean(prefix_covs[:, b]), alpha)
                         for b in range(prefix_covs.shape[1])])
