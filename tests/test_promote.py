"""Quality-gate `decide()` — pure logic, no MLflow."""
from __future__ import annotations

from churn_dbx.promote import decide

GATE = {"min_roc_auc": 0.58, "min_ap_lift_over_base": 1.10, "max_roc_auc_regression": 0.02}


def test_first_model_passes_on_baseline_checks():
    ok, reason = decide(0.62, 1.25, None, GATE)
    assert ok and "passed" in reason


def test_fails_below_auc_floor():
    ok, reason = decide(0.55, 1.25, None, GATE)
    assert not ok and "min_roc_auc" in reason


def test_fails_below_ap_lift():
    ok, reason = decide(0.62, 1.05, None, GATE)
    assert not ok and "min_ap_lift_over_base" in reason


def test_blocks_regression_against_champion():
    # 0.60 < champion 0.65 - 0.02 -> regression -> blocked (rollback by non-promotion).
    ok, reason = decide(0.60, 1.25, 0.65, GATE)
    assert not ok and "no_regression" in reason


def test_allows_small_dip_within_tolerance():
    ok, _ = decide(0.62, 1.25, 0.63, GATE)
    assert ok
