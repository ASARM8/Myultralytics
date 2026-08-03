# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Proposal-level modules for rotated, spatially aligned OBB refinement."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("OBBProposalRefinerV3",)


def _group_norm(channels: int) -> nn.GroupNorm:
    """Return a padding-independent normalization with a valid group count."""
    groups = next(group for group in (8, 4, 2, 1) if channels % group == 0)
    return nn.GroupNorm(groups, channels)


def _canonical_long_angle(width: torch.Tensor, height: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Map equivalent OBB representations to one undirected long-axis angle in [0, pi)."""
    long_angle = torch.where(width <= height, angle + math.pi / 2.0, angle)
    return torch.remainder(long_angle, math.pi)


class OBBProposalRefinerV3(nn.Module):
    """Refine post-NMS OBB proposals using P2/P3 rotated-aligned strips.

    This module is deliberately independent of the legacy dense ``OBBRefine``
    and ``OBBRefineV2`` heads.  It consumes a bounded number of proposal boxes,
    preserves the base detector scores/classes, and returns four residual slots:

    ``[delta_short, delta_long, delta_center_long, delta_center_short]``.

    The evidence-backed default trains scale only and keeps both center slots at
    zero. Center prediction remains an explicit opt-in for later ablation. The
    final geometry layer is zero initialized, so a newly constructed module
    is an exact identity mapping.  A separate quality logit is returned rather
    than silently changing the detector's classification score.
    """

    residual_names = ("dshort", "dlong", "dcenter_long", "dcenter_short")
    state_names = (
        "confidence",
        "log_short_norm",
        "log_long_norm",
        "log_aspect_ratio",
        "center_x_norm",
        "center_y_norm",
        "sin_2long_axis",
        "cos_2long_axis",
    )

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
        short_negative_limit: float = 1.5,
        short_positive_limit: float = 0.25,
        long_negative_limit: float = 0.15,
        long_positive_limit: float = 0.15,
        target_margin: float = 0.99,
        enable_center: bool = False,
        center_limit: float = 1.0,
    ) -> None:
        super().__init__()
        if min(p2_channels, p3_channels, roi_channels, hidden_channels) <= 0:
            raise ValueError("feature and hidden channel counts must be positive")
        if len(roi_size) != 2 or min(roi_size) <= 0:
            raise ValueError(f"roi_size must contain two positive integers, received {roi_size}")
        if min(
            long_context,
            short_context,
            min_short_context_px,
            short_negative_limit,
            short_positive_limit,
            long_negative_limit,
            long_positive_limit,
            center_limit,
        ) <= 0:
            raise ValueError("context sizes and residual limits must be positive")
        if not 0.0 < target_margin < 1.0:
            raise ValueError("target_margin must be strictly between 0 and 1")

        self.roi_size = (int(roi_size[0]), int(roi_size[1]))
        self.long_context = float(long_context)
        self.short_context = float(short_context)
        self.min_short_context_px = float(min_short_context_px)
        self.short_negative_limit = float(short_negative_limit)
        self.short_positive_limit = float(short_positive_limit)
        self.long_negative_limit = float(long_negative_limit)
        self.long_positive_limit = float(long_positive_limit)
        self.target_margin = float(target_margin)
        self.enable_center = bool(enable_center)
        self.center_limit = float(center_limit)

        self.p2_projection = nn.Sequential(
            nn.Conv2d(p2_channels, roi_channels, 1, bias=False),
            _group_norm(roi_channels),
            nn.SiLU(),
        )
        self.p3_projection = nn.Sequential(
            nn.Conv2d(p3_channels, roi_channels, 1, bias=False),
            _group_norm(roi_channels),
            nn.SiLU(),
        )
        combined_channels = 2 * roi_channels
        self.roi_encoder = nn.Sequential(
            nn.Conv2d(combined_channels, hidden_channels, 3, padding=1, bias=False),
            _group_norm(hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=(1, 2), padding=1, bias=False),
            _group_norm(hidden_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 6)),
        )
        state_channels = len(self.state_names)
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(state_channels),
            nn.Linear(state_channels, hidden_channels),
            nn.SiLU(),
        )
        fused_channels = hidden_channels * 2 * 6 + hidden_channels
        self.fusion = nn.Sequential(
            nn.Linear(fused_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
        )
        self.geometry_channels = len(self.residual_names) if self.enable_center else 2
        self.geometry_head = nn.Linear(hidden_channels, self.geometry_channels)
        self.quality_head = nn.Linear(hidden_channels, 1)
        nn.init.zeros_(self.geometry_head.weight)
        nn.init.zeros_(self.geometry_head.bias)
        nn.init.zeros_(self.quality_head.weight)
        nn.init.zeros_(self.quality_head.bias)
        self.register_buffer("_refine_v3_marker", torch.tensor(3, dtype=torch.int8), persistent=True)

    @staticmethod
    def _validate_inputs(
        p2: torch.Tensor,
        p3: torch.Tensor,
        proposals: torch.Tensor,
        scores: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> None:
        if p2.ndim != 4 or p3.ndim != 4:
            raise ValueError("P2 and P3 must have shape [B, C, H, W]")
        if proposals.ndim != 3 or proposals.shape[-1] != 5:
            raise ValueError("proposals must have shape [B, K, 5] in xywhr format")
        if scores.shape != proposals.shape[:2]:
            raise ValueError("scores must have shape [B, K]")
        if p2.shape[0] != proposals.shape[0] or p3.shape[0] != proposals.shape[0]:
            raise ValueError("feature maps and proposals must have the same batch size")
        if valid_mask is not None and valid_mask.shape != proposals.shape[:2]:
            raise ValueError("valid_mask must have shape [B, K]")

    def _rotated_grid(self, proposals: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        """Create an align-corners-false sampling grid packed as K strips per image."""
        image_height, image_width = int(image_size[0]), int(image_size[1])
        if image_height <= 0 or image_width <= 0:
            raise ValueError(f"image_size must be positive, received {image_size}")
        batch, proposal_count = proposals.shape[:2]
        roi_height, roi_width = self.roi_size
        width, height, angle = proposals[..., 2], proposals[..., 3], proposals[..., 4]
        short = torch.minimum(width, height)
        long = torch.maximum(width, height)
        long_angle = _canonical_long_angle(width, height, angle)
        long_extent = long * self.long_context
        short_extent = torch.maximum(
            short * self.short_context,
            torch.full_like(short, self.min_short_context_px),
        )
        u = torch.linspace(-0.5, 0.5, roi_width, device=proposals.device, dtype=proposals.dtype)
        v = torch.linspace(-0.5, 0.5, roi_height, device=proposals.device, dtype=proposals.dtype)
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        local_long = uu[None, None] * long_extent[..., None, None]
        local_short = vv[None, None] * short_extent[..., None, None]
        cosine = long_angle.cos()[..., None, None]
        sine = long_angle.sin()[..., None, None]
        x = proposals[..., 0, None, None] + local_long * cosine - local_short * sine
        y = proposals[..., 1, None, None] + local_long * sine + local_short * cosine
        grid = torch.stack((2.0 * x / image_width - 1.0, 2.0 * y / image_height - 1.0), dim=-1)
        return grid.reshape(batch, proposal_count * roi_height, roi_width, 2)

    def _sample(self, feature: torch.Tensor, grid: torch.Tensor, proposal_count: int) -> torch.Tensor:
        sampled = F.grid_sample(feature, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        batch, channels, _, roi_width = sampled.shape
        roi_height = self.roi_size[0]
        return sampled.reshape(batch, channels, proposal_count, roi_height, roi_width).permute(0, 2, 1, 3, 4)

    @staticmethod
    def proposal_state(proposals: torch.Tensor, scores: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        """Build low-dimensional state features used as a non-spatial baseline."""
        image_height, image_width = int(image_size[0]), int(image_size[1])
        normalizer = float(max(image_height, image_width))
        width, height, angle = proposals[..., 2], proposals[..., 3], proposals[..., 4]
        short = torch.minimum(width, height).clamp_min(1e-3)
        long = torch.maximum(width, height).clamp_min(1e-3)
        long_angle = _canonical_long_angle(width, height, angle)
        return torch.stack(
            (
                scores,
                torch.log(short / normalizer),
                torch.log(long / normalizer),
                torch.log(long / short),
                proposals[..., 0] / image_width,
                proposals[..., 1] / image_height,
                torch.sin(2.0 * long_angle),
                torch.cos(2.0 * long_angle),
            ),
            dim=-1,
        )

    @staticmethod
    def _asymmetric_bound(raw: torch.Tensor, negative_limit: float, positive_limit: float) -> torch.Tensor:
        """Apply sign-dependent tanh limits with value 0 and derivative 1 at the origin."""
        negative = float(negative_limit) * torch.tanh(raw / float(negative_limit))
        positive = float(positive_limit) * torch.tanh(raw / float(positive_limit))
        return torch.where(raw < 0, negative, positive)

    def bound_residual(self, raw: torch.Tensor) -> torch.Tensor:
        """Return bounded scale residuals and optional center residuals."""
        if raw.shape[-1] != self.geometry_channels:
            raise ValueError(f"expected {self.geometry_channels} raw geometry channels, received {raw.shape[-1]}")
        short = self._asymmetric_bound(raw[..., 0:1], self.short_negative_limit, self.short_positive_limit)
        long = self._asymmetric_bound(raw[..., 1:2], self.long_negative_limit, self.long_positive_limit)
        if self.enable_center:
            center = self.center_limit * torch.tanh(raw[..., 2:4] / self.center_limit)
        else:
            center = raw.new_zeros((*raw.shape[:-1], 2))
        return torch.cat((short, long, center), dim=-1)

    def clip_target(self, target: torch.Tensor) -> torch.Tensor:
        """Clip encoded supervision to exactly the residual range available at inference."""
        if target.shape[-1] != len(self.residual_names):
            raise ValueError(f"expected {len(self.residual_names)} target channels, received {target.shape[-1]}")
        # tanh only approaches its exact bound asymptotically. Keeping targets
        # just inside that range avoids an unreachable endpoint and vanishing
        # gradients for clipped examples.
        short = target[..., 0:1].clamp(
            -self.short_negative_limit * self.target_margin,
            self.short_positive_limit * self.target_margin,
        )
        long = target[..., 1:2].clamp(
            -self.long_negative_limit * self.target_margin,
            self.long_positive_limit * self.target_margin,
        )
        if self.enable_center:
            center = target[..., 2:4].clamp(
                -self.center_limit * self.target_margin,
                self.center_limit * self.target_margin,
            )
        else:
            center = target.new_zeros((*target.shape[:-1], 2))
        return torch.cat((short, long, center), dim=-1)

    def forward(
        self,
        p2: torch.Tensor,
        p3: torch.Tensor,
        proposals: torch.Tensor,
        scores: torch.Tensor,
        image_size: tuple[int, int],
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return bounded geometry residuals and independent quality logits."""
        self._validate_inputs(p2, p3, proposals, scores, valid_mask)
        proposal_count = proposals.shape[1]
        if proposal_count == 0:
            empty_residual = proposals.new_zeros((*proposals.shape[:2], len(self.residual_names)))
            return {"residual": empty_residual, "quality_logit": proposals.new_zeros((*proposals.shape[:2], 1))}
        grid = self._rotated_grid(proposals, image_size)
        p2_crop = self._sample(self.p2_projection(p2), grid, proposal_count)
        p3_crop = self._sample(self.p3_projection(p3), grid, proposal_count)
        crop = torch.cat((p2_crop, p3_crop), dim=2).flatten(0, 1)
        spatial = self.roi_encoder(crop).flatten(1)
        state = self.proposal_state(proposals, scores, image_size).flatten(0, 1)
        state = self.state_encoder(state)
        fused = self.fusion(torch.cat((spatial, state), dim=1))
        raw = self.geometry_head(fused).reshape(*proposals.shape[:2], self.geometry_channels)
        quality_logit = self.quality_head(fused).reshape(*proposals.shape[:2], 1)
        residual = self.bound_residual(raw)
        if valid_mask is not None:
            mask = valid_mask.to(device=residual.device, dtype=torch.bool)
            residual = residual.masked_fill(~mask[..., None], 0.0)
            quality_logit = quality_logit.masked_fill(~mask[..., None], -20.0)
        return {"residual": residual, "quality_logit": quality_logit}

    @staticmethod
    def encode_targets(proposals: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Encode aligned GT boxes as V3 short/long scale and local-center targets."""
        if proposals.shape != targets.shape or proposals.shape[-1] != 5:
            raise ValueError("proposals and targets must have the same [..., 5] shape")
        proposal_short = proposals[..., 2:4].amin(dim=-1).clamp_min(1e-3)
        proposal_long = proposals[..., 2:4].amax(dim=-1).clamp_min(1e-3)
        target_short = targets[..., 2:4].amin(dim=-1).clamp_min(1e-3)
        target_long = targets[..., 2:4].amax(dim=-1).clamp_min(1e-3)
        scale = torch.stack(
            (torch.log(target_short / proposal_short), torch.log(target_long / proposal_long)),
            dim=-1,
        )
        long_angle = _canonical_long_angle(proposals[..., 2], proposals[..., 3], proposals[..., 4])
        difference = targets[..., :2] - proposals[..., :2]
        long_offset = difference[..., 0] * long_angle.cos() + difference[..., 1] * long_angle.sin()
        short_offset = -difference[..., 0] * long_angle.sin() + difference[..., 1] * long_angle.cos()
        center = torch.stack(
            (long_offset / proposal_long, short_offset / proposal_short.clamp_min(4.0)),
            dim=-1,
        )
        return torch.cat((scale, center), dim=-1)

    @staticmethod
    def apply_residual(proposals: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Apply V3 residuals without changing proposal angle, class, or score."""
        if proposals.shape[:-1] != residual.shape[:-1] or proposals.shape[-1] != 5 or residual.shape[-1] != 4:
            raise ValueError("expected proposals [..., 5] and residual [..., 4] with aligned leading dimensions")
        refined = proposals.clone()
        short_is_width = proposals[..., 2] <= proposals[..., 3]
        delta_width = torch.where(short_is_width, residual[..., 0], residual[..., 1])
        delta_height = torch.where(short_is_width, residual[..., 1], residual[..., 0])
        refined[..., 2] *= torch.exp(delta_width)
        refined[..., 3] *= torch.exp(delta_height)
        short = proposals[..., 2:4].amin(dim=-1).clamp_min(4.0)
        long = proposals[..., 2:4].amax(dim=-1).clamp_min(1e-3)
        long_angle = _canonical_long_angle(proposals[..., 2], proposals[..., 3], proposals[..., 4])
        long_offset = residual[..., 2] * long
        short_offset = residual[..., 3] * short
        refined[..., 0] += long_offset * long_angle.cos() - short_offset * long_angle.sin()
        refined[..., 1] += long_offset * long_angle.sin() + short_offset * long_angle.cos()
        return refined
