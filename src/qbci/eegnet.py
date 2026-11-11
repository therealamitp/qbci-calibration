"""Compact EEGNet-style raw EEG comparator with constrained fine-tuning.

Independent implementation of the depthwise/separable pattern, not an exact
replication of a published checkpoint. Batch-normalization statistics freeze
during calibration and evaluation. Importing this module requires PyTorch.
"""
import copy
import numpy as np
import torch
from torch import nn
from .protocol import domain_weights
from .preprocessing import causal_filter_epochs


class EEGNet(nn.Module):
    def __init__(self, channels, samples, temporal_kernel=64, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, (1, temporal_kernel), padding=(0, temporal_kernel//2), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            nn.Conv2d(16, 16, (1, 16), padding=(0, 8), groups=16, bias=False),
            nn.Conv2d(16, 16, (1, 1), bias=False), nn.BatchNorm2d(16),
            nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        self.features.eval()
        with torch.no_grad():
            dimensions = self.features(torch.zeros(1, 1, channels, samples)).numel()
        self.head = nn.Linear(dimensions, 2)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class EEGNetAdapter:
    def __init__(self, seed=0, source_epochs=100, target_epochs=20, lr=1e-3,
                 anchor=0.01, target_mass=0.5, batch_size=32):
        self.seed, self.source_epochs, self.target_epochs = seed, source_epochs, target_epochs
        self.lr, self.anchor, self.target_mass, self.batch_size = lr, anchor, target_mass, batch_size

    def _raw(self, epochs):
        filtered = causal_filter_epochs(epochs, self.sfreq_, bands=((8, 30),))[:, 0]
        return filtered[:, :, int(0.5*self.sfreq_):int(3.5*self.sfreq_)]

    def _tensor(self, epochs):
        z = (self._raw(epochs)-self.mean_)/self.std_
        return torch.as_tensor(z[:, None], dtype=torch.float32)

    def fit_source(self, epochs, y, sfreq):
        torch.manual_seed(self.seed)
        self.sfreq_ = sfreq
        raw = self._raw(epochs)
        self.mean_ = raw.mean(axis=(0, 2), keepdims=True)
        self.std_ = np.maximum(raw.std(axis=(0, 2), keepdims=True), 1e-10)
        self.x_source_, self.y_source_ = self._tensor(epochs), torch.as_tensor(y, dtype=torch.long)
        self.model_ = EEGNet(raw.shape[1], raw.shape[2], temporal_kernel=max(16, int(0.5*sfreq)))
        self._train(self.x_source_, self.y_source_, np.ones(len(y))/len(y), self.source_epochs, False)
        self.source_state_ = copy.deepcopy(self.model_.state_dict())
        return self

    def _train(self, x, y, weights, epochs, adapting):
        parameters = [p for p in self.model_.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=self.lr, weight_decay=1e-3)
        reference = {n: p.detach().clone() for n, p in self.model_.named_parameters()}
        generator = torch.Generator().manual_seed(self.seed)
        weights = torch.as_tensor(weights, dtype=torch.float32)
        for _ in range(epochs):
            self.model_.train()
            if adapting:
                self.model_.features.eval()
            order = torch.randperm(len(y), generator=generator)
            for start in range(0, len(y), self.batch_size):
                idx = order[start:start+self.batch_size]
                logits = self.model_(x[idx])
                losses = nn.functional.cross_entropy(logits, y[idx], reduction="none")
                loss = (losses*weights[idx]).sum()*len(y)/len(idx)
                if adapting:
                    penalty = sum((p-reference[n]).pow(2).mean() for n, p in self.model_.named_parameters() if p.requires_grad)
                    loss = loss + self.anchor*penalty
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, 5.0)
                optimizer.step()
        self.model_.eval()

    def adapt(self, epochs, y, mode="classifier"):
        self.model_.load_state_dict(self.source_state_)
        if mode == "frozen" or len(y) == 0:
            self.model_.eval()
            return self
        if mode != "classifier":
            raise ValueError("Only constrained head adaptation is implemented")
        for name, p in self.model_.named_parameters():
            p.requires_grad_(name.startswith("head."))
        x = torch.cat([self.x_source_, self._tensor(epochs)])
        labels = torch.cat([self.y_source_, torch.as_tensor(y, dtype=torch.long)])
        weights = domain_weights(len(self.y_source_), len(y), self.target_mass)
        self._train(x, labels, weights, self.target_epochs, True)
        return self

    def predict(self, epochs):
        self.model_.eval()
        with torch.no_grad():
            return self.model_(self._tensor(epochs)).argmax(dim=1).cpu().numpy()
