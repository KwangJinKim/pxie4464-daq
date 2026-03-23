from __future__ import annotations
import logging
from collections import deque

import numpy as np
from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from pxie4464_daq.analysis.fft import compute_fft
from pxie4464_daq.analysis.features import extract_features

logger = logging.getLogger(__name__)

N_CHANNELS = 4


class FeatureCollector(QObject):
    """주기적으로 4채널 FFT 특징을 추출하여 emit.

    Signals:
        features_ready(object): shape (4, 7) numpy 배열
    """

    features_ready = pyqtSignal(object)

    def __init__(self, sample_rate: float, collection_cycle_sec: float = 30.0,
                 window_sec: float = 5.0, parent=None):
        super().__init__(parent)
        self._sample_rate = sample_rate
        self._window_samples = int(sample_rate * window_sec)
        # 채널별 rolling buffer (deque로 자동 truncation)
        self._buffers = [deque(maxlen=self._window_samples) for _ in range(N_CHANNELS)]
        self._timer = QTimer(self)
        self._timer.setInterval(int(collection_cycle_sec * 1000))
        self._timer.timeout.connect(self._extract_and_emit)

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def on_data_ready(self, data: np.ndarray) -> None:
        """AcquisitionWorker.data_ready 시그널 슬롯. data: (4, N)"""
        for ch in range(N_CHANNELS):
            self._buffers[ch].extend(data[ch].tolist())

    def _extract_and_emit(self) -> None:
        features_all = np.zeros((N_CHANNELS, 7), dtype=np.float64)
        for ch in range(N_CHANNELS):
            if len(self._buffers[ch]) < 2:
                logger.warning("CH%d: 버퍼 부족 (%d 샘플)", ch, len(self._buffers[ch]))
                continue
            chunk = np.array(self._buffers[ch], dtype=np.float64)
            freqs, mags = compute_fft(chunk, self._sample_rate)
            features_all[ch] = extract_features(freqs, mags)
        self.features_ready.emit(features_all)
