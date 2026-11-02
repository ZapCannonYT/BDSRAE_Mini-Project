"""Retraining controller: train a challenger, gate it, promote it or keep the champion."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from taxidrift.eval.metrics import wape
from taxidrift.model.features import FeatureSpec
from taxidrift.model.train import predict, train_model


@dataclass
class RetrainResult:
    promoted: bool
    challenger_wape: float
    champion_wape: float
    seconds: float


class RetrainController:
    def __init__(self, spec: FeatureSpec, train_window_weeks: int, gate_holdout_hours: int, rounds: int = 100,
                 num_threads: int = 0, log_model=None) -> None:
        self.spec = spec
        self.train_hours = train_window_weeks * 168
        self.holdout = gate_holdout_hours
        self.rounds = rounds
        self.num_threads = num_threads
        self.log_model = log_model  # optional callback(model, result, reason), e.g. MLflow
        self.champion = None

    def retrain(self, t: int, Y: np.ndarray, hours: pd.DatetimeIndex, reason: str) -> RetrainResult:
        """Train on the recent window ending before the holdout; promote only if better on the holdout.

        t is the index of the latest observed hour. The holdout hours are never used for fitting.
        """
        start = time.perf_counter()
        end = t + 1
        hold = np.arange(end - self.holdout, end)
        train = np.arange(max(self.spec.max_history, end - self.holdout - self.train_hours), end - self.holdout)
        challenger = train_model(Y, hours, train, self.spec, rounds=self.rounds, num_threads=self.num_threads)
        y = Y[hold]
        ch = wape(y, predict(challenger, Y, hours, hold, self.spec, self.num_threads))
        cm = wape(y, predict(self.champion, Y, hours, hold, self.spec, self.num_threads))
        promoted = ch < cm
        result = RetrainResult(promoted, ch, cm, time.perf_counter() - start)
        if promoted:
            self.champion = challenger
        if self.log_model:
            self.log_model(challenger, result, reason)
        return result
