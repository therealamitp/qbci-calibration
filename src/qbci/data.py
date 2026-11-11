"""Typed sessions and synthetic motor-band data for software verification."""
from dataclasses import dataclass
import json
import numpy as np


@dataclass
class Session:
    dataset: str
    subject: str
    session: str
    session_order: int
    epochs: np.ndarray
    labels: np.ndarray
    sfreq: float
    channels: tuple
    trial_ids: np.ndarray
    feedback: bool = False

    def __post_init__(self):
        self.epochs = np.asarray(self.epochs, float)
        self.labels = np.asarray(self.labels, int)
        self.trial_ids = np.asarray(self.trial_ids)
        self.channels = tuple(self.channels)
        if self.epochs.ndim != 3 or len(self.epochs) != len(self.labels):
            raise ValueError("Invalid epoch/label shapes")
        if len(self.channels) != self.epochs.shape[1] or len(self.trial_ids) != len(self.labels):
            raise ValueError("Invalid channels or trial identifiers")
        if len(np.unique(self.trial_ids)) != len(self.trial_ids):
            raise ValueError("Trial identifiers must be unique within session")
        if set(np.unique(self.labels)) - {0, 1} or not np.isfinite(self.epochs).all():
            raise ValueError("Invalid labels or nonfinite EEG")

    def save(self, path):
        meta = {k: getattr(self, k) for k in ("dataset", "subject", "session", "session_order",
                                             "sfreq", "channels", "feedback")}
        np.savez_compressed(path, epochs=self.epochs, labels=self.labels,
                            trial_ids=self.trial_ids, metadata=json.dumps(meta))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(**json.loads(str(data["metadata"])), epochs=data["epochs"],
                       labels=data["labels"], trial_ids=data["trial_ids"])


def synthetic_pair(seed=0, trials=80, channels=6, sfreq=128.0):
    """Software smoke fixture, never evidence for quantum EEG advantage."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sfreq*4))/sfreq
    mixing = np.eye(channels) + rng.normal(0, 0.12, (channels, channels))
    sessions = []
    for session in range(2):
        labels = rng.permutation(np.arange(trials) % 2)
        x = rng.normal(0, 0.7, (trials, channels, len(t)))
        for i, label in enumerate(labels):
            x[i, label] += 1.2*np.sin(2*np.pi*10*t+rng.uniform(0, 2*np.pi))
            x[i, 1-label] += 0.7*np.sin(2*np.pi*20*t+rng.uniform(0, 2*np.pi))
        if session:
            x = np.einsum("ab,nbt->nat", mixing, x) + rng.normal(0, 0.18, x.shape)
        sessions.append(Session("SYNTHETIC", str(seed), str(session+1), session+1,
                                x, labels, sfreq, tuple("C%d" % i for i in range(channels)),
                                np.arange(trials)))
    return tuple(sessions)
