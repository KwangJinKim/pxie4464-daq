from __future__ import annotations
import numpy as np

HARMONIC_EXCLUSION_BINS = 5  # 주도 주파수 ±5 bin 제외 (노이즈 플로어 계산용)


def extract_features(frequencies: np.ndarray, magnitudes: np.ndarray) -> np.ndarray:
    """FFT 스펙트럼에서 7개 특징 추출.

    Returns:
        np.ndarray shape (7,):
            [0] dominant_freq_hz
            [1] dominant_magnitude
            [2] second_harmonic_magnitude
            [3] third_harmonic_magnitude
            [4] thd (simplified: (H2+H3)/H1)
            [5] noise_floor_rms
            [6] spectral_centroid_hz
    """
    # DC 제거 후 주도 주파수 탐색 (index 0 제외)
    search_mags = magnitudes.copy()
    search_mags[0] = 0.0
    dom_idx = int(np.argmax(search_mags))
    dom_freq = frequencies[dom_idx]
    dom_mag = magnitudes[dom_idx]

    # 하모닉 크기 추출 (해당 주파수 bin에서)
    def _harmonic_mag(n: int) -> float:
        target = dom_freq * n
        if target > frequencies[-1]:
            return 0.0
        idx = int(np.argmin(np.abs(frequencies - target)))
        return float(magnitudes[idx])

    h2 = _harmonic_mag(2)
    h3 = _harmonic_mag(3)
    thd = (h2 + h3) / dom_mag if dom_mag > 0 else 0.0

    # 노이즈 플로어 RMS: 주도 주파수 ±5 bin 마스킹 후 계산
    mask = np.ones(len(magnitudes), dtype=bool)
    lo = max(0, dom_idx - HARMONIC_EXCLUSION_BINS)
    hi = min(len(magnitudes), dom_idx + HARMONIC_EXCLUSION_BINS + 1)
    mask[lo:hi] = False
    noise_rms = float(np.sqrt(np.mean(magnitudes[mask] ** 2))) if mask.any() else 0.0

    # 스펙트럼 센트로이드
    total_mag = np.sum(magnitudes)
    centroid = float(np.dot(frequencies, magnitudes) / total_mag) if total_mag > 0 else 0.0

    return np.array([dom_freq, dom_mag, h2, h3, thd, noise_rms, centroid], dtype=np.float64)
