import numpy as np
import pytest
from pxie4464_daq.analysis.fft import compute_fft
from pxie4464_daq.analysis.features import extract_features

SR = 51200.0
N = 4096


def make_sine(freq: float, amplitude: float = 1.0) -> tuple:
    t = np.arange(N) / SR
    data = amplitude * np.sin(2 * np.pi * freq * t)
    return compute_fft(data, SR)


def test_extract_features_returns_7_elements():
    freqs, mags = make_sine(100.0)
    features = extract_features(freqs, mags)
    assert features.shape == (7,)
    assert features.dtype == np.float64


def test_dominant_frequency():
    freqs, mags = make_sine(200.0)
    features = extract_features(freqs, mags)
    dominant_freq = features[0]
    assert abs(dominant_freq - 200.0) < 5.0


def test_dominant_magnitude():
    freqs, mags = make_sine(100.0, amplitude=2.0)
    features = extract_features(freqs, mags)
    dominant_mag = features[1]
    assert abs(dominant_mag - 2.0) < 0.05


def test_thd_pure_sine_is_low():
    """순수 사인파의 THD는 매우 낮아야 함"""
    freqs, mags = make_sine(100.0)
    features = extract_features(freqs, mags)
    thd = features[4]
    assert thd < 0.05  # 5% 이하


def test_spectral_centroid_near_dominant():
    """단일 주파수 신호의 스펙트럼 센트로이드는 주도 주파수 근처"""
    freqs, mags = make_sine(300.0)
    features = extract_features(freqs, mags)
    centroid = features[6]
    assert abs(centroid - 300.0) < 50.0
