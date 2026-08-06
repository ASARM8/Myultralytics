"""Torch-free unit tests for V3.1.1 reproduction comparisons."""

from myscripts.V3_1_1.audit_reproductions_v311 import metric_differences


def test_metric_differences_covers_both_variants_and_all_threshold_metrics():
    left = {
        "coarse": {"map50_95": 0.4, "ap75": 0.3, "ap90": 0.2, "ap95": 0.1},
        "refined": {"map50_95": 0.5, "ap75": 0.4, "ap90": 0.3, "ap95": 0.2},
    }
    right = {
        "coarse": {"map50_95": 0.39, "ap75": 0.29, "ap90": 0.19, "ap95": 0.09},
        "refined": {"map50_95": 0.48, "ap75": 0.38, "ap90": 0.28, "ap95": 0.18},
    }
    differences = metric_differences(left, right)
    assert set(differences) == {
        f"{variant}.{metric}"
        for variant in ("coarse", "refined")
        for metric in ("map50_95", "ap75", "ap90", "ap95")
    }
    assert abs(differences["coarse.map50_95"] - 0.01) < 1e-12
    assert abs(differences["refined.ap95"] - 0.02) < 1e-12
