"""Refine V3.1.1 with smoothly compressed conservative supervision."""

from __future__ import annotations

import torch

from ultralytics.nn.modules.refine_v31 import OBBProposalRefinerV31


class OBBProposalRefinerV311(OBBProposalRefinerV31):
    """Geometry-only V3.1 head with targets kept away from physical bounds.

    The physical inference limits remain identical to V3.1.  Only the target
    mapping changes: exact log-scale targets are smoothly compressed into
    ``supervision_margin`` times each physical limit.  This preserves unit
    derivative at zero while avoiding the point mass created by hard clipping
    37% of short-side targets at 99% of the output boundary.
    """

    def __init__(
        self,
        p2_channels: int,
        p3_channels: int,
        *,
        roi_channels: int = 32,
        roi_size: tuple[int, int] = (5, 24),
        hidden_channels: int = 128,
        long_context: float = 1.2,
        short_context: float = 4.0,
        min_short_context_px: float = 16.0,
        short_negative_limit: float = 0.50,
        short_positive_limit: float = 0.20,
        long_negative_limit: float = 0.08,
        long_positive_limit: float = 0.08,
        supervision_margin: float = 0.80,
    ) -> None:
        if not 0.0 < supervision_margin < 1.0:
            raise ValueError("supervision_margin must be strictly between 0 and 1")
        super().__init__(
            p2_channels=p2_channels,
            p3_channels=p3_channels,
            roi_channels=roi_channels,
            roi_size=roi_size,
            hidden_channels=hidden_channels,
            long_context=long_context,
            short_context=short_context,
            min_short_context_px=min_short_context_px,
            short_negative_limit=short_negative_limit,
            short_positive_limit=short_positive_limit,
            long_negative_limit=long_negative_limit,
            long_positive_limit=long_positive_limit,
            target_margin=supervision_margin,
            use_quality_aux=False,
        )
        self.supervision_margin = float(supervision_margin)
        self.register_buffer("_refine_v311_marker", torch.tensor(311, dtype=torch.int16), persistent=True)

    @staticmethod
    def _smooth_target(raw: torch.Tensor, negative_limit: float, positive_limit: float) -> torch.Tensor:
        """Smoothly compress signed targets with value and derivative continuous at zero."""
        negative = float(negative_limit) * torch.tanh(raw / float(negative_limit))
        positive = float(positive_limit) * torch.tanh(raw / float(positive_limit))
        return torch.where(raw < 0, negative, positive)

    def clip_target(self, target: torch.Tensor) -> torch.Tensor:
        """Map targets inside the conservative supervision range without hard clipping."""
        if target.shape[-1] != len(self.residual_names):
            raise ValueError(f"expected {len(self.residual_names)} target channels, received {target.shape[-1]}")
        short = self._smooth_target(
            target[..., 0:1],
            self.short_negative_limit * self.supervision_margin,
            self.short_positive_limit * self.supervision_margin,
        )
        long = self._smooth_target(
            target[..., 1:2],
            self.long_negative_limit * self.supervision_margin,
            self.long_positive_limit * self.supervision_margin,
        )
        center = target.new_zeros((*target.shape[:-1], 2))
        return torch.cat((short, long, center), dim=-1)
