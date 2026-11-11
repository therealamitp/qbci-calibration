"""Full-feature and spatial classical controls, all source or prefix fitted."""
import numpy as np
from scipy.linalg import eigh
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .preprocessing import covariance_epochs
from .protocol import domain_weights, representation_update_allowed


class WeightedLDA:
    """LDA with weighted moments, equal class priors, and covariance shrinkage."""
    def __init__(self, shrinkage=0.1):
        self.shrinkage = shrinkage

    def fit(self, x, y, weights=None):
        weights = np.ones(len(y)) if weights is None else np.asarray(weights)
        means, covs = [], []
        for c in (0, 1):
            selected = y == c
            if not selected.any():
                raise ValueError("LDA needs both classes, use source replay")
            w = weights[selected] / weights[selected].sum()
            mean = w @ x[selected]
            residual = x[selected] - mean
            means.append(mean)
            covs.append((residual.T * w) @ residual)
        covariance = (covs[0]+covs[1])/2
        scale = max(np.trace(covariance)/len(covariance), 1e-10)
        covariance = (1-self.shrinkage)*covariance + (self.shrinkage+1e-8)*scale*np.eye(len(covariance))
        self.coef_ = np.linalg.solve(covariance, means[1]-means[0])
        self.intercept_ = -0.5*(means[1]+means[0]) @ self.coef_
        return self

    def predict(self, x):
        return (x @ self.coef_ + self.intercept_ > 0).astype(int)


class FBCSP:
    def __init__(self, bands=((8, 12), (12, 16), (16, 20), (20, 24), (24, 30)), pairs=2):
        self.bands, self.pairs = bands, pairs

    def fit(self, epochs, y, sfreq):
        self.sfreq_ = sfreq
        covs = covariance_epochs(epochs, sfreq, self.bands)
        self.filters_ = []
        for band in range(len(self.bands)):
            c0, c1 = (covs[y == c, band].mean(axis=0) for c in (0, 1))
            _, vectors = eigh(c1, c0+c1)
            count = min(self.pairs, len(c0)//2)
            self.filters_.append(vectors[:, np.r_[np.arange(count), np.arange(len(c0)-count, len(c0))]])
        self.scaler_ = StandardScaler().fit(self._features(covs))
        return self

    def _features(self, covs):
        columns = []
        for band, w in enumerate(self.filters_):
            variances = np.einsum("ia,nij,ja->na", w, covs[:, band], w)
            columns.append(np.log(np.maximum(variances, 1e-20)))
        return np.concatenate(columns, axis=1)

    def transform(self, epochs):
        c = covariance_epochs(epochs, self.sfreq_, self.bands)
        return self.scaler_.transform(self._features(c))


class TangentProcrustes:
    """Regularized centroid Procrustes in a source-fitted low-dimensional space.

    This is a deliberately specified TSA-style control, NOT an exact replication
    of Bleuze et al.'s clustering procedure. Full-dimensional RA is also required.
    """
    def __init__(self, dimensions=16, regularization=1.0):
        self.dimensions, self.regularization = dimensions, regularization

    def fit_source(self, x, y):
        self.pca_ = PCA(n_components=min(self.dimensions, min(x.shape)-1)).fit(x)
        self.source_ = self.pca_.transform(x)
        self.labels_ = y.copy()
        self.rotation_ = np.eye(self.source_.shape[1])
        return self

    def adapt(self, x, y):
        self.rotation_ = np.eye(self.source_.shape[1])
        if len(x) == 0:
            return self
        target = self.pca_.transform(x)
        if representation_update_allowed(y):
            a = np.stack([target[y == c].mean(axis=0) for c in (0, 1)])
            b = np.stack([self.source_[self.labels_ == c].mean(axis=0) for c in (0, 1)])
            u, _, vt = np.linalg.svd(a.T @ b + self.regularization*self.rotation_)
            self.rotation_ = u @ vt
        return self

    def transform(self, x):
        if len(x) == 0:
            return np.empty((0, self.source_.shape[1]))
        return self.pca_.transform(x) @ self.rotation_


def fit_replay_classifier(xs, ys, xt, yt, kind="svm", C=1.0, gamma=1.0, target_mass=0.5):
    x, y = np.concatenate([xs, xt]), np.concatenate([ys, yt])
    weights = domain_weights(len(xs), len(xt), target_mass)
    if kind == "lda":
        return WeightedLDA().fit(x, y, weights)
    return SVC(C=C, gamma=gamma).fit(x, y, sample_weight=weights*len(xs))
