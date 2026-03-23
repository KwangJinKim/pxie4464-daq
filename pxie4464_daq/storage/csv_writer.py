from __future__ import annotations
import csv
import os
from datetime import datetime
from typing import List

import numpy as np


def save_raw(data: np.ndarray, sample_rate: float, timestamp: datetime,
             output_dir: str = ".") -> None:
    """4채널 원시 가속도 데이터를 채널별 CSV로 저장.

    Args:
        data: shape (4, N) 가속도 배열 (g)
        sample_rate: 샘플레이트 (S/s), 시간 축 계산용
        timestamp: 파일명에 사용할 타임스탬프
        output_dir: 저장 디렉토리
    """
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    n_samples = data.shape[1]
    time_arr = np.arange(n_samples) / sample_rate
    os.makedirs(output_dir, exist_ok=True)
    for ch in range(data.shape[0]):
        path = os.path.join(output_dir, f"raw_{ts_str}_ch{ch}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "acceleration_g"])
            for t, v in zip(time_arr, data[ch]):
                writer.writerow([f"{t:.8f}", f"{v:.8f}"])


def save_fft(frequencies: List[np.ndarray], magnitudes: List[np.ndarray],
             timestamp: datetime, output_dir: str = ".") -> None:
    """4채널 FFT 스펙트럼을 채널별 CSV로 저장.

    Args:
        frequencies: 채널별 주파수 배열 리스트 (길이 4)
        magnitudes: 채널별 크기 배열 리스트 (길이 4)
        timestamp: 파일명에 사용할 타임스탐프
        output_dir: 저장 디렉토리
    """
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    for ch, (freqs, mags) in enumerate(zip(frequencies, magnitudes)):
        path = os.path.join(output_dir, f"fft_{ts_str}_ch{ch}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frequency_hz", "magnitude"])
            for freq, mag in zip(freqs, mags):
                writer.writerow([f"{freq:.4f}", f"{mag:.8f}"])
