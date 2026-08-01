"""CPU-only tests for Refine V3 probe-audit argument helpers."""

import argparse

import pytest

from myscripts.refine_v3_probe_audit import parse_float_csv, stable_seed


def test_parse_float_csv_accepts_nonnegative_grid():
    assert parse_float_csv("0,1e-3,0.1,1") == (0.0, 0.001, 0.1, 1.0)


@pytest.mark.parametrize("value", ["", "-0.1,1", "abc"])
def test_parse_float_csv_rejects_invalid_grid(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_float_csv(value)


def test_stable_seed_is_reproducible_and_context_sensitive():
    first = stable_seed(0, "scale", "all_positive", "P3")
    assert first == stable_seed(0, "scale", "all_positive", "P3")
    assert first != stable_seed(0, "angle", "all_positive", "P3")
