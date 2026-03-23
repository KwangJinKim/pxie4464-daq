from __future__ import annotations
import logging
from datetime import datetime
from typing import Optional, List

import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox,
    QComboBox, QMessageBox, QGridLayout
)
from PyQt5.QtCore import Qt

from pxie4464_daq.device.daq import MockDAQ, _DAQBase
from pxie4464_daq.acquisition.worker import AcquisitionWorker
from pxie4464_daq.analysis.fft import compute_fft
from pxie4464_daq.analysis.feature_collector import FeatureCollector
from pxie4464_daq.analysis.anomaly_detector import AnomalyDetector
from pxie4464_daq.storage.csv_writer import save_raw, save_fft
from pxie4464_daq.ui.waveform_plot import WaveformPlot
from pxie4464_daq.ui.fft_plot import FFTPlot
from pxie4464_daq.ui.anomaly_plot import AnomalyPlot
from pxie4464_daq.ui.status_light import StatusLight

logger = logging.getLogger(__name__)

VOLTAGE_RANGES = [1.0, 3.16, 10.0, 31.6]


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PXIe-4464 진동 데이터 수집")
        self._daq: Optional[_DAQBase] = None
        self._worker: Optional[AcquisitionWorker] = None
        self._collector: Optional[FeatureCollector] = None
        self._detector: Optional[AnomalyDetector] = None
        self._last_data: Optional[np.ndarray] = None
        self._last_freqs: Optional[List] = None
        self._last_mags: Optional[List] = None
        self._setup_ui()

    # ── UI 구성 ─────────────────────────────────────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # 좌측: 제어 패널
        ctrl = self._make_control_panel()
        root.addWidget(ctrl, stretch=1)

        # 우측: 플롯 영역
        plots = QVBoxLayout()
        top_plots = QHBoxLayout()
        self._waveform_plot = WaveformPlot()
        self._fft_plot = FFTPlot()
        top_plots.addWidget(self._waveform_plot)
        top_plots.addWidget(self._fft_plot)
        plots.addLayout(top_plots)

        bottom_plots = QHBoxLayout()
        self._anomaly_plot = AnomalyPlot()
        self._status_light = StatusLight()
        bottom_plots.addWidget(self._anomaly_plot, stretch=3)
        bottom_plots.addWidget(self._status_light, stretch=1)
        plots.addLayout(bottom_plots)

        root.addLayout(plots, stretch=4)

    def _make_control_panel(self) -> QGroupBox:
        group = QGroupBox("설정")
        layout = QGridLayout(group)
        row = 0

        layout.addWidget(QLabel("장치명"), row, 0)
        self._device_name_edit = QLineEdit("Dev1")
        layout.addWidget(self._device_name_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("샘플레이트 (S/s)"), row, 0)
        self._sample_rate_edit = QLineEdit("51200")
        layout.addWidget(self._sample_rate_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("청크 크기 (샘플)"), row, 0)
        self._chunk_edit = QLineEdit("1024")
        layout.addWidget(self._chunk_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("전압 범위 (±V)"), row, 0)
        self._voltage_combo = QComboBox()
        for v in VOLTAGE_RANGES:
            self._voltage_combo.addItem(f"±{v}V", v)
        self._voltage_combo.setCurrentIndex(2)  # ±10V 기본
        layout.addWidget(self._voltage_combo, row, 1)
        row += 1

        self._mock_check = QCheckBox("Mock 모드")
        self._mock_check.setChecked(True)
        layout.addWidget(self._mock_check, row, 0, 1, 2)
        row += 1

        self._connect_btn = QPushButton("연결")
        self._connect_btn.clicked.connect(self._on_connect)
        layout.addWidget(self._connect_btn, row, 0, 1, 2)
        row += 1

        self._start_btn = QPushButton("▶ 시작")
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._on_start_stop)
        layout.addWidget(self._start_btn, row, 0, 1, 2)
        row += 1

        self._save_btn = QPushButton("CSV 저장")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save_csv)
        layout.addWidget(self._save_btn, row, 0, 1, 2)

        return group

    # ── 슬롯 ────────────────────────────────────────────────────────────────

    def _on_connect(self):
        try:
            sample_rate = float(self._sample_rate_edit.text())
            chunk = int(self._chunk_edit.text())
            voltage_range = self._voltage_combo.currentData()

            if self._mock_check.isChecked():
                self._daq = MockDAQ()
            else:
                from pxie4464_daq.device.daq import PXIe4464
                self._daq = PXIe4464(device_name=self._device_name_edit.text())

            self._daq.configure(sample_rate=sample_rate, record_length=chunk,
                                voltage_range=voltage_range)

            self._waveform_plot._sample_rate = sample_rate
            self._collector = FeatureCollector(sample_rate=sample_rate)
            self._detector = AnomalyDetector()
            self._detector.state_changed.connect(self._on_state_changed)
            self._collector.features_ready.connect(self._detector.update)

            self._start_btn.setEnabled(True)
            self._connect_btn.setEnabled(False)
            logger.info("DAQ 연결 완료: %s, sr=%.0f, chunk=%d",
                        type(self._daq).__name__, sample_rate, chunk)
        except Exception as exc:
            QMessageBox.critical(self, "연결 오류", str(exc))

    def _on_start_stop(self):
        if self._worker is None or not self._worker.isRunning():
            self._start_acquisition()
        else:
            self._stop_acquisition()

    def _start_acquisition(self):
        self._worker = AcquisitionWorker(self._daq)
        self._worker.data_ready.connect(self._on_data_ready)
        self._worker.error_occurred.connect(self._on_error)
        self._collector.start()
        self._worker.start()
        self._start_btn.setText("■ 정지")
        self._save_btn.setEnabled(True)

    def _stop_acquisition(self):
        if self._worker:
            self._worker.stop()
            self._worker = None
        if self._collector:
            self._collector.stop()
        self._start_btn.setText("▶ 시작")

    def _on_data_ready(self, data: np.ndarray):
        self._last_data = data
        self._waveform_plot.update(data)

        # 실시간 FFT 갱신
        sample_rate = float(self._sample_rate_edit.text())
        freqs_list, mags_list = [], []
        for ch in range(4):
            freqs, mags = compute_fft(data[ch], sample_rate)
            freqs_list.append(freqs)
            mags_list.append(mags)
        self._last_freqs = freqs_list
        self._last_mags = mags_list
        self._fft_plot.update(freqs_list, mags_list)

        # FeatureCollector에 데이터 전달
        self._collector.on_data_ready(data)

    def _on_state_changed(self, states):
        self._status_light.update_states(states)
        if self._detector:
            self._anomaly_plot.update(self._detector.if_scores())

    def _on_error(self, msg: str):
        QMessageBox.critical(self, "수집 오류", msg)
        self._stop_acquisition()

    def _on_save_csv(self):
        if self._last_data is None:
            QMessageBox.warning(self, "저장 실패", "저장할 데이터가 없습니다.")
            return
        ts = datetime.now()
        sample_rate = float(self._sample_rate_edit.text())
        try:
            save_raw(self._last_data, sample_rate=sample_rate, timestamp=ts)
            if self._last_freqs and self._last_mags:
                save_fft(self._last_freqs, self._last_mags, timestamp=ts)
            QMessageBox.information(self, "저장 완료", f"CSV 저장 완료: {ts.strftime('%Y%m%d_%H%M%S')}")
        except Exception as exc:
            QMessageBox.critical(self, "저장 오류", str(exc))

    def closeEvent(self, event):
        self._stop_acquisition()
        super().closeEvent(event)
