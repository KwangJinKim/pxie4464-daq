from __future__ import annotations
import numpy as np
from typing import Tuple


def compute_fft(data: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """1D 가속도 데이터에 Hanning 윈도우 FFT를 적용하여 단측 스펙트럼을 반환.

    Args:
        data: shape (N,) 1D 가속도 배열 (g 단위)
        sample_rate: 샘플레이트 (S/s)

    Returns:
        (frequencies, magnitudes): 각각 shape (M,) ndarray, M = N//2 + 1
            magnitudes는 진폭 보정된 값 (g 단위)
    """
    N = len(data)
    window = np.hanning(N)
    windowed = data * window
    spectrum = np.fft.rfft(windowed)
    # Hanning 윈도우 진폭 보정: 2.0/sum(window) 사용 (2.0/N 대비 ~50% 오차 방지)
    magnitudes = np.abs(spectrum) * 2.0 / np.sum(window)
    frequencies = np.fft.rfftfreq(N, d=1.0 / sample_rate)
    return frequencies, magnitudes
