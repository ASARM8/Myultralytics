"""Conservative proposal-level OBB refinement used by Refine V3.1."""

from __future__ import annotations

import torch

from ultralytics.nn.modules.refine_v3 import OBBProposalRefinerV3


class OBBProposalRefinerV31(OBBProposalRefinerV3):
    """Refine every valid CA proposal without an inference gate or a second NMS.

    V3.1 retains the rotated P2/P3 ROI encoder and the short/long log-scale
    residual representation from V3.  It narrows the residual bounds and makes
    the quality head optional.  When enabled, quality is training-only metadata;
    it never controls inference.
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
        target_margin: float = 0.99,
        use_quality_aux: bool = False,
    ) -> None:
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
            target_margin=target_margin,
            enable_center=False,
            center_limit=1.0,
        )
        self.use_quality_aux = bool(use_quality_aux)
        if not self.use_quality_aux:
            # Removing the module also removes its parameters from the optimizer
            # and checkpoint, making geometry_only a genuine smaller model.
            self.quality_head = None
        self.register_buffer("_refine_v31_marker", torch.tensor(31, dtype=torch.int8), persistent=True)

    def forward(
        self,
        p2: torch.Tensor,
        p3: torch.Tensor,
        proposals: torch.Tensor,
        scores: torch.Tensor,
        image_size: tuple[int, int],
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Return bounded geometry residuals and an optional auxiliary logit."""
        self._validate_inputs(p2, p3, proposals, scores, valid_mask)
        proposal_count = proposals.shape[1]
        if proposal_count == 0:
            empty_residual = proposals.new_zeros((*proposals.shape[:2], len(self.residual_names)))
            empty_quality = proposals.new_zeros((*proposals.shape[:2], 1)) if self.use_quality_aux else None
            return {"residual": empty_residual, "quality_logit": empty_quality}

        grid = self._rotated_grid(proposals, image_size)
        p2_crop = self._sample(self.p2_projection(p2), grid, proposal_count)
        p3_crop = self._sample(self.p3_projection(p3), grid, proposal_count)
        crop = torch.cat((p2_crop, p3_crop), dim=2).flatten(0, 1)
        spatial = self.roi_encoder(crop).flatten(1)
        state = self.state_encoder(self.proposal_state(proposals, scores, image_size).flatten(0, 1))
        fused = self.fusion(torch.cat((spatial, state), dim=1))

        raw = self.geometry_head(fused).reshape(*proposals.shape[:2], self.geometry_channels)
        residual = self.bound_residual(raw)
        quality_logit = None
        if self.quality_head is not None:
            quality_logit = self.quality_head(fused).reshape(*proposals.shape[:2], 1)

        if valid_mask is not None:
            mask = valid_mask.to(device=residual.device, dtype=torch.bool)
            residual = residual.masked_fill(~mask[..., None], 0.0)
            if quality_logit is not None:
                quality_logit = quality_logit.masked_fill(~mask[..., None], -20.0)
        return {"residual": residual, "quality_logit": quality_logit}
