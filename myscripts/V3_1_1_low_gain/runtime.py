"""Shared read-only runtime helpers for the isolated low-gain evidence chain."""

from __future__ import annotations

from typing import Any


class ResidualScaledRefiner:
    """Scale only the decoded geometry residual returned by a frozen refiner.

    Attribute access is delegated to the wrapped module so the official
    evaluator, profiler, and qualitative exporter retain the same checkpoint,
    parameters, bounds, and ``apply_residual`` implementation.
    """

    def __init__(self, refiner: Any, residual_scale: float) -> None:
        self._refiner = refiner
        self.residual_scale = float(residual_scale)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._refiner, name)

    def eval(self) -> "ResidualScaledRefiner":
        self._refiner.eval()
        return self

    def train(self, mode: bool = True) -> "ResidualScaledRefiner":
        self._refiner.train(mode)
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        output = self._refiner(*args, **kwargs)
        if not isinstance(output, dict) or "residual" not in output:
            raise TypeError("V3.1.1 refiner did not return a residual dictionary")
        scaled = dict(output)
        scaled["residual"] = output["residual"] * self.residual_scale
        return scaled


def require_scale(parser, value: float) -> None:
    """Validate a generic sensitivity coefficient."""
    if not 0.0 <= float(value) <= 1.0:
        parser.error("residual-scale must be within [0, 1]")
