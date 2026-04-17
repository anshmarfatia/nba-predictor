"""Tests for the calibration measurement + recalibration wrappers.

Calibration shrinks ECE — that's the whole point. The tests build a
deliberately miscalibrated probability stream (shifted sigmoid) and check
that both Isotonic and Platt recalibration push ECE down. Save/load
round-trips are also checked since the calibrators are serialized to JSON
alongside model artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.models.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    expected_calibration_error,
    load_calibrator,
    reliability_table,
)


def _miscalibrated_stream(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate (y_true, y_raw_prob) where y_raw_prob is systematically too
    confident relative to the true labels — a classic calibration failure mode."""
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.1, 0.9, size=n)
    y = (rng.uniform(size=n) < true_p).astype(int)
    # Inflate toward the extremes: quadratic distortion.
    raw = np.clip(true_p ** 2 / (true_p ** 2 + (1 - true_p) ** 2), 1e-4, 1 - 1e-4)
    return y, raw


def test_reliability_table_shape_and_bounds():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.6, 0.2, 0.95])
    tbl = reliability_table(y, p, n_bins=5)
    assert len(tbl) == 5
    assert tbl["n"].sum() == len(y)
    assert (tbl["bin_low"] >= 0).all() and (tbl["bin_high"] <= 1).all()


def test_ece_is_zero_when_perfectly_calibrated():
    # Probabilities equal the empirical rate exactly within each bin.
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    p = np.array([0.5] * 8)
    ece = expected_calibration_error(reliability_table(y, p, n_bins=10))
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_isotonic_reduces_ece():
    y, raw = _miscalibrated_stream()
    raw_ece = expected_calibration_error(reliability_table(y, raw))
    cal = IsotonicCalibrator().fit(y, raw)
    calibrated = cal.transform(raw)
    cal_ece = expected_calibration_error(reliability_table(y, calibrated))
    assert cal_ece < raw_ece


def test_platt_reduces_ece():
    y, raw = _miscalibrated_stream()
    raw_ece = expected_calibration_error(reliability_table(y, raw))
    cal = PlattCalibrator().fit(y, raw)
    calibrated = cal.transform(raw)
    cal_ece = expected_calibration_error(reliability_table(y, calibrated))
    assert cal_ece < raw_ece


def test_isotonic_save_load_round_trip(tmp_path: Path):
    y, raw = _miscalibrated_stream(n=1000)
    cal = IsotonicCalibrator().fit(y, raw)
    path = tmp_path / "iso.json"
    cal.save(path)

    loaded = IsotonicCalibrator.load(path)
    original = cal.transform(raw)
    restored = loaded.transform(raw)
    np.testing.assert_allclose(restored, original, atol=1e-9)

    # Tag is preserved so the generic loader can dispatch.
    state = json.loads(path.read_text())
    assert state["method"] == "isotonic"


def test_platt_save_load_round_trip(tmp_path: Path):
    y, raw = _miscalibrated_stream(n=1000)
    cal = PlattCalibrator().fit(y, raw)
    path = tmp_path / "platt.json"
    cal.save(path)

    loaded = PlattCalibrator.load(path)
    np.testing.assert_allclose(loaded.transform(raw), cal.transform(raw), atol=1e-12)

    state = json.loads(path.read_text())
    assert state["method"] == "platt"


def test_generic_loader_dispatches_by_method(tmp_path: Path):
    y, raw = _miscalibrated_stream(n=500)
    iso_path = tmp_path / "a.json"
    platt_path = tmp_path / "b.json"
    IsotonicCalibrator().fit(y, raw).save(iso_path)
    PlattCalibrator().fit(y, raw).save(platt_path)

    assert isinstance(load_calibrator(iso_path), IsotonicCalibrator)
    assert isinstance(load_calibrator(platt_path), PlattCalibrator)


def test_unfitted_calibrator_cannot_transform_or_save(tmp_path: Path):
    with pytest.raises(RuntimeError):
        IsotonicCalibrator().transform([0.5])
    with pytest.raises(RuntimeError):
        IsotonicCalibrator().save(tmp_path / "x.json")
    with pytest.raises(RuntimeError):
        PlattCalibrator().transform([0.5])
    with pytest.raises(RuntimeError):
        PlattCalibrator().save(tmp_path / "y.json")


def test_calibrated_probs_stay_in_unit_interval():
    y, raw = _miscalibrated_stream(n=500)
    for cal in (IsotonicCalibrator().fit(y, raw), PlattCalibrator().fit(y, raw)):
        out = cal.transform(np.array([0.0, 0.001, 0.5, 0.999, 1.0]))
        assert (out >= 0).all() and (out <= 1).all()
