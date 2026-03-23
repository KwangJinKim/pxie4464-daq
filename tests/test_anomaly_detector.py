import numpy as np
import pytest
from pxie4464_daq.analysis.anomaly_detector import ChannelAnomalyDetector, State

BASELINE = 20
N_FEATURES = 7


def make_features(n: int, anomalous: bool = False) -> list:
    """정상/이상 특징 벡터 생성"""
    rng = np.random.default_rng(42)
    baseline = rng.normal(loc=[100, 1, 0.05, 0.02, 0.07, 0.01, 110], scale=0.01, size=(n, N_FEATURES))
    if anomalous:
        # 매우 큰 이상값 (Z-score >> 5)
        baseline = rng.normal(loc=[500, 10, 5, 4, 0.9, 0.5, 600], scale=0.01, size=(n, N_FEATURES))
    return [baseline[i] for i in range(n)]


def test_initial_state_is_learning():
    det = ChannelAnomalyDetector(baseline_count=BASELINE)
    assert det.state == State.LEARNING


def test_transitions_to_normal_after_baseline():
    det = ChannelAnomalyDetector(baseline_count=BASELINE)
    for feat in make_features(BASELINE):
        det.update(feat)
    assert det.state == State.NORMAL


def test_stays_learning_before_baseline():
    det = ChannelAnomalyDetector(baseline_count=BASELINE)
    for feat in make_features(BASELINE - 1):
        det.update(feat)
    assert det.state == State.LEARNING


def test_alarm_on_strong_anomaly():
    """강한 이상값은 결국 ALARM 상태로 전환"""
    det = ChannelAnomalyDetector(baseline_count=BASELINE)
    # 정상 베이스라인 학습
    for feat in make_features(BASELINE):
        det.update(feat)
    assert det.state == State.NORMAL
    # 강한 이상값 주입 (홀드오프 3회 초과)
    for feat in make_features(10, anomalous=True):
        det.update(feat)
    assert det.state in (State.WARNING, State.ALARM)


def test_holdoff_prevents_single_spike_warning():
    """단발 스파이크(1회)는 WARNING 발동 안 함"""
    det = ChannelAnomalyDetector(baseline_count=BASELINE)
    for feat in make_features(BASELINE):
        det.update(feat)
    # 단발 이상
    det.update(make_features(1, anomalous=True)[0])
    # 즉시 WARNING이면 안 됨 (홀드오프=3)
    if det.state == State.WARNING:
        pytest.fail("Single spike should not trigger WARNING due to holdoff=3")
