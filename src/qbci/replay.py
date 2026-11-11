"""Separate prequential analysis. This is not the fixed-suffix primary curve."""
import numpy as np
from .protocol import LabelOracle


def sequential_replay(source_model, target_features, target_labels, budget=32,
                      mode="subset", update_every=4):
    """Reveal the first `budget` labels, always AFTER prediction.

    Parameters are refit from the source anchor at each update. Every observation
    is scored, including the calibration period. No active/drift selection is
    silently introduced. Returned truth is for analysis after predictions only.
    """
    if budget < 0 or budget > len(target_features):
        raise ValueError("Invalid label budget")
    oracle = LabelOracle(target_labels, budget)
    revealed_x, revealed_y, rows = [], [], []
    empty = np.empty((0, target_features.shape[1]))
    source_model.adapt(empty, np.empty(0, int), "classifier")
    for i, x in enumerate(target_features):
        prediction = int(source_model.predict(x[None])[0])
        oracle.score(i, prediction)
        before = len(oracle.revealed)
        if i < budget:
            label = oracle.reveal(i)
            revealed_x.append(x.copy())
            revealed_y.append(label)
            if len(revealed_y) % update_every == 0 or len(revealed_y) == budget:
                source_model.adapt(np.asarray(revealed_x), np.asarray(revealed_y), mode)
        rows.append({"trial": i, "prediction": prediction,
                     "labels_available_at_prediction": before,
                     "label_revealed_after_prediction": i < budget})
    for row, truth in zip(rows, target_labels):
        row["truth"] = int(truth)
    return rows, oracle.events
