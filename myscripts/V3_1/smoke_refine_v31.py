"""Run one Refine V3.1 train batch before launching a full cloud run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .train_refine_v31 import CANONICAL_CA_WEIGHTS, EXPERIMENTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test one V3.1 forward/backward step on train only.")
    parser.add_argument("--refiner-version", choices=("v31", "v311"), default="v31", help=argparse.SUPPRESS)
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    parser.add_argument("--ca-weights", default=CANONICAL_CA_WEIGHTS)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640, choices=(640,))
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if Path(args.ca_weights).as_posix() != Path(CANONICAL_CA_WEIGHTS).as_posix():
        parser.error(f"V3.1 smoke test is locked to the canonical CA checkpoint: {CANONICAL_CA_WEIGHTS}")
    if args.batch <= 0 or args.workers < 0:
        parser.error("batch must be positive and workers must be non-negative")
    if args.refiner_version == "v311" and args.experiment != "geometry_only":
        parser.error("Refine V3.1.1 smoke test is fixed to --experiment geometry_only")
    if not os.environ.get("OMP_NUM_THREADS", "").isdigit() or int(os.environ.get("OMP_NUM_THREADS", "0")) <= 0:
        os.environ["OMP_NUM_THREADS"] = "1"

    import torch
    import torch.nn.functional as F

    from ultralytics import YOLO
    from ultralytics.nn.modules.refine_v31 import OBBProposalRefinerV31
    from ultralytics.nn.modules.refine_v311 import OBBProposalRefinerV311
    from ultralytics.utils.torch_utils import select_device

    from myscripts.V3.train_refine_v3 import focal_binary_loss

    from .runtime import FrozenCAExtractor, build_dataset, build_supervision, full_loader, pad_detections, sha256_file

    device = select_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    ca_hash_before = sha256_file(args.ca_weights)
    wrapper = YOLO(args.ca_weights, task="obb")
    ca_model = wrapper.model.to(device).float().eval()
    head = ca_model.model[-1]
    if type(head).__name__ != "OBB" or int(getattr(head, "reg_max", -1)) != 32:
        raise RuntimeError("smoke test requires the canonical pure CA OBB(reg_max=32) checkpoint")
    for parameter in ca_model.parameters():
        parameter.requires_grad_(False)

    dataset, data = build_dataset(args.data, "train", args.imgsz, args.batch, args.workers, rect=False)
    loader = full_loader(dataset, args.batch, args.workers, shuffle=False)
    extractor = FrozenCAExtractor(
        ca_model,
        device=device,
        nc=len(getattr(ca_model, "names", data["names"])),
        conf=0.01,
        nms_iou=0.70,
        max_det=300,
        amp=use_amp,
    )
    try:
        p2_channels, p3_channels = extractor.infer_channels(args.imgsz)
        use_quality_aux = args.experiment == "quality_aux"
        if args.refiner_version == "v311":
            refiner = OBBProposalRefinerV311(p2_channels, p3_channels, supervision_margin=0.80)
        else:
            refiner = OBBProposalRefinerV31(p2_channels, p3_channels, use_quality_aux=use_quality_aux)
        refiner = refiner.to(device).float().train()
        optimizer = torch.optim.AdamW(refiner.parameters(), lr=3e-4, weight_decay=1e-4)
        batch = next(iter(loader))
        images, p2, p3, detections = extractor.infer(batch)
        boxes, scores, classes, valid = pad_detections(detections)
        if not valid.any():
            raise RuntimeError("smoke-test batch has no post-NMS CA proposals")
        supervision = build_supervision(
            refiner,
            boxes.float(),
            classes,
            valid,
            batch,
            images.shape[2:],
            match_iou=0.30,
            quality_min_gain=0.002,
            tiny_reference_px=8.0,
            tiny_weight_floor=0.25,
        )
        if not supervision["matched"].any():
            raise RuntimeError("smoke-test batch has no class-aware matched proposal")
        before = {name: parameter.detach().clone() for name, parameter in refiner.named_parameters()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = refiner(p2, p3, boxes, scores, images.shape[2:], valid)
            if not torch.equal(output["residual"], torch.zeros_like(output["residual"])):
                raise RuntimeError("zero-initialized V3.1 did not start as geometry identity")
            matched = supervision["matched"]
            geometry_loss = F.smooth_l1_loss(
                output["residual"][matched][..., :2],
                supervision["clipped_target"][matched][..., :2],
                beta=0.05,
            )
            identity_mask = valid & ~supervision["quality_target"].bool()
            identity_loss = F.smooth_l1_loss(
                output["residual"][identity_mask][..., :2],
                torch.zeros_like(output["residual"][identity_mask][..., :2]),
                beta=0.05,
            ) if identity_mask.any() else output["residual"].sum() * 0.0
            if use_quality_aux:
                quality_logit = output["quality_logit"]
                if quality_logit is None:
                    raise RuntimeError("quality_aux smoke test has no quality output")
                quality_loss = focal_binary_loss(
                    torch,
                    quality_logit.squeeze(-1)[valid],
                    supervision["quality_target"][valid],
                    0.75,
                    2.0,
                )
            else:
                if output["quality_logit"] is not None:
                    raise RuntimeError("geometry_only unexpectedly created a quality output")
                quality_loss = output["residual"].sum() * 0.0
            loss = geometry_loss + 0.02 * identity_loss + (0.5 * quality_loss if use_quality_aux else 0.0)
        if not torch.isfinite(loss):
            raise RuntimeError("smoke-test loss is non-finite")
        loss.backward()
        if any(parameter.grad is not None for parameter in ca_model.parameters()):
            raise RuntimeError("frozen CA unexpectedly received gradients")
        optimizer.step()
        changed = [
            name for name, parameter in refiner.named_parameters() if not torch.equal(before[name], parameter.detach())
        ]
        if not changed:
            raise RuntimeError("optimizer step did not update any V3.1 parameter")
        ca_hash_after = sha256_file(args.ca_weights)
        if ca_hash_before != ca_hash_after:
            raise RuntimeError("canonical CA weight file changed during the V3.1 smoke test")
        result = {
            "status": "PASS",
            "architecture": type(refiner).__name__,
            "experiment": args.experiment,
            "split": "train",
            "val_used": False,
            "test_used": False,
            "batch_images": int(images.shape[0]),
            "valid_proposals": int(valid.sum()),
            "matched_proposals": int(supervision["matched"].sum()),
            "loss": float(loss.detach()),
            "geometry_loss": float(geometry_loss.detach()),
            "quality_loss": float(quality_loss.detach()),
            "identity_loss": float(identity_loss.detach()),
            "ca_sha256": ca_hash_after,
            "updated_parameter_tensors": changed,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(output_path)
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
