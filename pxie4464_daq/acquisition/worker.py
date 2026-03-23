from __future__ import annotations
import logging

from PyQt5.QtCore import QThread, pyqtSignal

from pxie4464_daq.device.daq import _DAQBase

logger = logging.getLogger(__name__)


class AcquisitionWorker(QThread):
    """백그라운드 연속 수집 스레드.

    Signals:
        data_ready(object): shape (4, N) numpy 배열
        error_occurred(str): 오류 메시지
    """

    data_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, daq: _DAQBase, parent=None):
        super().__init__(parent)
        self._daq = daq
        self._running = False

    def run(self) -> None:
        self._running = True
        try:
            self._daq.start()
            while self._running:
                data = self._daq.read()
                self.data_ready.emit(data)
        except Exception as exc:
            logger.error("AcquisitionWorker error: %s", exc)
            self.error_occurred.emit(str(exc))
        finally:
            try:
                self._daq.stop()
            except Exception as exc:
                logger.warning("DAQ stop error (ignored): %s", exc)

    def stop(self) -> None:
        self._running = False
        self.wait()
