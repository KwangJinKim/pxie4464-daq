import csv
import os
import numpy as np
import pytest
from datetime import datetime
from pxie4464_daq.storage.csv_writer import save_raw, save_fft


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


def test_save_raw_creates_files_per_channel(tmp_dir):
    data = np.random.randn(4, 512)
    ts = datetime(2026, 3, 23, 12, 0, 0)
    save_raw(data, sample_rate=51200.0, timestamp=ts, output_dir=str(tmp_dir))
    for ch in range(4):
        fname = tmp_dir / f"raw_20260323_120000_ch{ch}.csv"
        assert fname.exists(), f"{fname} not found"


def test_save_raw_correct_columns(tmp_dir):
    data = np.ones((4, 4))  # 4ch, 4 samples
    ts = datetime(2026, 3, 23, 0, 0, 0)
    save_raw(data, sample_rate=4.0, timestamp=ts, output_dir=str(tmp_dir))
    with open(tmp_dir / "raw_20260323_000000_ch0.csv") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert "time_s" in rows[0]
    assert "acceleration_g" in rows[0]
    assert len(rows) == 4


def test_save_fft_creates_files(tmp_dir):
    freqs = [np.linspace(0, 100, 50) for _ in range(4)]
    mags = [np.random.rand(50) for _ in range(4)]
    ts = datetime(2026, 3, 23, 9, 30, 0)
    save_fft(freqs, mags, timestamp=ts, output_dir=str(tmp_dir))
    for ch in range(4):
        assert (tmp_dir / f"fft_20260323_093000_ch{ch}.csv").exists()
