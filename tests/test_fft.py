import numpy as np
import pytest
from pxie4464_daq.analysis.fft import compute_fft


def test_compute_fft_shape():
    N = 1024
    sr = 51200.0
    data = np.zeros(N)
    freqs, mags = compute_fft(data, sr)
    assert len(freqs) == len(mags)
    assert freqs[0] >= 0
    assert freqs[-1] <= sr / 2


def test_compute_fft_recovers_amplitude():
    """Hanning 윈도우 정규화 후 1g 사인파 진폭이 1.0에 가까운지 확인 (오차 1% 이내)"""
    sr = 51200.0
    N = 4096
    freq = 100.0
    t = np.arange(N) / sr
    data = 1.0 * np.sin(2 * np.pi * freq * t)
    freqs, mags = compute_fft(data, sr)
    peak_idx = np.argmax(mags)
    assert abs(freqs[peak_idx] - freq) < 2.0, f"Peak at wrong freq: {freqs[peak_idx]}"
    assert abs(mags[peak_idx] - 1.0) < 0.01, f"Amplitude error: {mags[peak_idx]}"


def test_compute_fft_zero_input():
    N = 512
    sr = 1024.0
    freqs, mags = compute_fft(np.zeros(N), sr)
    assert np.allclose(mags, 0.0)
