"""Shared audited loading helpers for Refine V3.1.1 evidence tools.

The module deliberately keeps PyTorch and Ultralytics imports inside loader
functions so command-line ``--help`` and protocol unit tests remain usable on
machines that do not have the cloud training environment installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CANONICAL_BASELINE_WEIGHTS = "/root/autodl-tmp/work-dirs/yolo11_obb_640_811_baseline/weights/best.pt"
CANONICAL_CA_WEIGHTS = "/root/autodl-tmp/work-dirs/yolo11_obb_640_811_ca/weights/best.pt"
REQUIRED_ARCHITECTURE = "OBBProposalRefinerV311"


def ensure_omp_threads() -> None:
    """Set a valid conservative OpenMP thread count before numerical imports."""
    value = os.environ.get("OMP_NUM_THREADS", "")
    if not value.isdigit() or int(value) <= 0:
        os.environ["OMP_NUM_THREADS"] = "1"


def require_canonical_path(parser, value: str, canonical: str, label: str) -> None:
    """Reject accidental evidence collection with a similarly named checkpoint."""
    if Path(value).as_posix() != Path(canonical).as_posix():
        parser.error(f"{label} is locked to: {canonical}")


@dataclass
class RefineEvidenceBundle:
    """Loaded frozen CA detector, feature extractor, and V3.1.1 refiner."""

    torch: Any
    device: Any
    use_amp: bool
    ca_path: Path
    ca_hash: str
    ca_model: Any
    extractor: Any
    refiner: Any
    checkpoint_path: Path
    checkpoint_hash: str
    checkpoint: dict[str, Any]
    training_args: dict[str, Any]

    def close(self) -> None:
        self.extractor.close()


@dataclass
class DetectorEvidenceBundle:
    """Loaded frozen OBB detector and its common NMS extractor."""

    weights_path: Path
    weights_hash: str
    model: Any
    extractor: Any
    reg_max: int

    def close(self) -> None:
        self.extractor.close()


def load_refine_bundle(
    checkpoint: str | Path,
    ca_weights: str | Path,
    *,
    device_arg: str,
    amp: bool,
    imgsz: int,
) -> RefineEvidenceBundle:
    """Load and strictly validate the frozen CA + V3.1.1 evidence chain."""
    ensure_omp_threads()

    import torch

    from ultralytics import YOLO
    from ultralytics.nn.modules.refine_v311 import OBBProposalRefinerV311
    from ultralytics.utils.torch_utils import select_device

    from myscripts.V3.runtime import FrozenCAExtractor, sha256_file

    checkpoint_path = Path(checkpoint)
    ca_path = Path(ca_weights)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Refine V3.1.1 checkpoint not found: {checkpoint_path}")
    if not ca_path.is_file():
        raise FileNotFoundError(f"canonical CA checkpoint not found: {ca_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1 or payload.get("architecture") != REQUIRED_ARCHITECTURE:
        raise RuntimeError(
            f"expected {REQUIRED_ARCHITECTURE} format_version=1 checkpoint, "
            f"received architecture={payload.get('architecture')!r}, format={payload.get('format_version')!r}"
        )
    ca_hash = sha256_file(ca_path)
    if payload.get("ca_sha256") != ca_hash:
        raise RuntimeError("CA checkpoint hash mismatch; Refine features and proposals no longer match training")

    device = select_device(device_arg)
    use_amp = bool(amp and device.type == "cuda")
    yolo = YOLO(str(ca_path), task="obb")
    ca_model = yolo.model.to(device).float().eval()
    head = ca_model.model[-1]
    if type(head).__name__ != "OBB" or int(getattr(head, "reg_max", -1)) != 32:
        raise RuntimeError(
            "expected pure CA OBB(reg_max=32), received "
            f"{type(head).__name__}(reg_max={getattr(head, 'reg_max', None)})"
        )
    for parameter in ca_model.parameters():
        parameter.requires_grad_(False)

    training_args = dict(payload.get("arguments", {}))
    extractor = FrozenCAExtractor(
        ca_model,
        device=device,
        nc=len(getattr(ca_model, "names", {})),
        conf=float(training_args.get("proposal_conf", 0.01)),
        nms_iou=float(training_args.get("nms_iou", 0.70)),
        max_det=int(training_args.get("max_det", 300)),
        amp=use_amp,
    )
    try:
        config = payload.get("model_config")
        if not isinstance(config, dict):
            raise RuntimeError("Refine checkpoint does not contain model_config")
        observed_channels = extractor.infer_channels(imgsz)
        expected_channels = int(config["p2_channels"]), int(config["p3_channels"])
        if observed_channels != expected_channels:
            raise RuntimeError(f"CA feature-channel mismatch: expected {expected_channels}, got {observed_channels}")
        refiner = OBBProposalRefinerV311(**config).to(device).float().eval()
        refiner.load_state_dict(payload["model_state"], strict=True)
    except Exception:
        extractor.close()
        raise

    return RefineEvidenceBundle(
        torch=torch,
        device=device,
        use_amp=use_amp,
        ca_path=ca_path,
        ca_hash=ca_hash,
        ca_model=ca_model,
        extractor=extractor,
        refiner=refiner,
        checkpoint_path=checkpoint_path,
        checkpoint_hash=sha256_file(checkpoint_path),
        checkpoint=payload,
        training_args=training_args,
    )


def load_obb_detector(
    weights: str | Path,
    *,
    device: Any,
    amp: bool,
    conf: float,
    nms_iou: float,
    max_det: int,
    expected_reg_max: int,
) -> DetectorEvidenceBundle:
    """Load an OBB detector under the same NMS protocol as the Refine chain."""
    from ultralytics import YOLO

    from myscripts.V3.runtime import FrozenCAExtractor, sha256_file

    weights_path = Path(weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"OBB detector checkpoint not found: {weights_path}")
    yolo = YOLO(str(weights_path), task="obb")
    model = yolo.model.to(device).float().eval()
    head = model.model[-1]
    observed_reg_max = int(getattr(head, "reg_max", -1))
    if type(head).__name__ != "OBB" or observed_reg_max != int(expected_reg_max):
        raise RuntimeError(
            f"expected OBB(reg_max={expected_reg_max}), received "
            f"{type(head).__name__}(reg_max={getattr(head, 'reg_max', None)})"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    extractor = FrozenCAExtractor(
        model,
        device=device,
        nc=len(getattr(model, "names", {})),
        conf=float(conf),
        nms_iou=float(nms_iou),
        max_det=int(max_det),
        amp=bool(amp),
    )
    return DetectorEvidenceBundle(
        weights_path=weights_path,
        weights_hash=sha256_file(weights_path),
        model=model,
        extractor=extractor,
        reg_max=observed_reg_max,
    )
