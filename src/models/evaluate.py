"""Classification metrics for binary win-probability models.

Accuracy alone is misleading for betting — a 62%-accurate model with badly
miscalibrated probabilities can be worse than a 60% model with good ones.
We always report accuracy + log loss + Brier + ROC-AUC together.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


@dataclass(frozen=True)
class Metrics:
    n: int
    accuracy: float
    log_loss: float
    brier: float
    roc_auc: float

    def as_dict(self) -> dict:
        return asdict(self)

    def pretty(self) -> str:
        return (
            f"n={self.n:<5}  acc={self.accuracy:.4f}  "
            f"log_loss={self.log_loss:.4f}  brier={self.brier:.4f}  "
            f"auc={self.roc_auc:.4f}"
        )


def evaluate(y_true: Iterable[int], y_prob: Iterable[float]) -> Metrics:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    y_pred = (y_prob >= 0.5).astype(int)

    return Metrics(
        n=len(y_true),
        accuracy=accuracy_score(y_true, y_pred),
        log_loss=log_loss(y_true, y_prob),
        brier=brier_score_loss(y_true, y_prob),
        roc_auc=roc_auc_score(y_true, y_prob),
    )
