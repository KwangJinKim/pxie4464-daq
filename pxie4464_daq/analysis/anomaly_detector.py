from __future__ import annotations
import logging
from enum import Enum, auto
from typing import List

import numpy as np
from sklearn.ensemble import IsolationForest
from PyQt5.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

ZSCORE_WARNING = 3.0
ZSCORE_ALARM = 5.0
IF_WARNING = -0.05   # decision_function() 임계값 (경험적 조정 필요)
IF_ALARM = -0.15
WARNING_HOLDOFF = 3  # 연속 n회 이상 판정 시에만 WARNING 발동


class State(Enum):
    LEARNING = auto()
    NORMAL = auto()
    WARNING = auto()
    ALARM = auto()


class ChannelAnomalyDetector:
    """단일 채널 이상 감지기."""

    def __init__(self, baseline_count: int = 20):
        self._baseline_count = baseline_count
        self._baseline: list = []
        self._model: IsolationForest | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self.state: State = State.LEARNING
        self._if_score: float = 0.0
        self._zscore_max: float = 0.0
        self._warning_streak: int = 0

    def update(self, features: np.ndarray) -> State:
        if self.state == State.LEARNING:
            self._baseline.append(features.copy())
            if len(self._baseline) >= self._baseline_count:
                self._fit_model()
                self.state = State.NORMAL
            return self.state

        # Z-score 계산
        zscores = np.abs((features - self._mean) / (self._std + 1e-10))
        self._zscore_max = float(np.max(zscores))

        # IsolationForest decision_function 점수
        self._if_score = float(self._model.decision_function(features.reshape(1, -1))[0])

        # 상태 결정 (OR 로직, 더 심각한 쪽 기준)
        if self._zscore_max >= ZSCORE_ALARM or self._if_score <= IF_ALARM:
            new_state = State.ALARM
        elif self._zscore_max >= ZSCORE_WARNING or self._if_score <= IF_WARNING:
            new_state = State.WARNING
        else:
            new_state = State.NORMAL

        # 홀드오프: WARNING/ALARM 3회 연속이어야 발동
        if new_state in (State.WARNING, State.ALARM):
            self._warning_streak += 1
        else:
            self._warning_streak = 0

        if self._warning_streak >= WARNING_HOLDOFF:
            self.state = new_state
        else:
            self.state = State.NORMAL

        return self.state

    @property
    def if_score(self) -> float:
        return self._if_score

    @property
    def zscore_max(self) -> float:
        return self._zscore_max

    def _fit_model(self) -> None:
        data = np.array(self._baseline)
        self._mean = data.mean(axis=0)
        self._std = data.std(axis=0)
        self._model = IsolationForest(contamination=0.05, random_state=42)
        self._model.fit(data)


class AnomalyDetector(QObject):
    """4채널 통합 이상 감지 관리자."""

    state_changed = pyqtSignal(object)  # list[State] (4채널)

    def __init__(self, baseline_count: int = 20, parent=None):
        super().__init__(parent)
        self._detectors = [ChannelAnomalyDetector(baseline_count) for _ in range(4)]

    def update(self, features: np.ndarray) -> List[State]:
        """features: shape (4, 7)"""
        states = [self._detectors[ch].update(features[ch]) for ch in range(4)]
        self.state_changed.emit(states)
        return states

    def if_scores(self) -> List[float]:
        return [d.if_score for d in self._detectors]

    def zscore_maxes(self) -> List[float]:
        return [d.zscore_max for d in self._detectors]
