"""Export aligned GT/Baseline/CA/CA+Refine predictions for qualitative figures."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Any

from myscripts.V3.train_refine_v3 import write_csv, write_json
from myscripts.V3_1_1.evidence_runtime import (
    CANONICAL_BASELINE_WEIGHTS,
    CANONICAL_CA_WEIGHTS,
    load_obb_detector,
    load_refine_bundle,
    require_canonical_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-weights", default=CANONICAL_BASELINE_WEIGHTS)
    parser.add_argument("--ca-weights", default=CANONICAL_CA_WEIGHTS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val", choices=("val",))
    parser.add_argument("--imgsz", type=int, default=640, choices=(640,))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-images", type=int, default=0, help="0 exports the complete validation split")
    parser.add_argument(
        "--copy-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy source images so the downloaded manifest remains portable",
    )
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    require_canonical_path(parser, args.baseline_weights, CANONICAL_BASELINE_WEIGHTS, "Baseline checkpoint")
    require_canonical_path(parser, args.ca_weights, CANONICAL_CA_WEIGHTS, "CA checkpoint")
    if args.batch <= 0 or args.workers < 0 or args.max_images < 0:
        parser.error("batch must be positive; workers/max-images must be non-negative")
    if args.amp:
        parser.error("official qualitative export is locked to FP32; use --no-amp")


def stable_export_id(image_path: str | Path) -> str:
    """Create a readable collision-resistant identifier for one source image."""
    path = Path(image_path)
    digest = hashlib.sha256(path.as_posix().encode("utf-8")).hexdigest()[:12]
    safe_stem = "".join(character if character.isalnum() or character in "-_" else "_" for character in path.stem)
    return f"{safe_stem}__{digest}"


def _scale_rboxes(ops, boxes, input_shape, original_shape, ratio_pad):
    if not boxes.shape[0]:
        return boxes.clone()
    return ops.scale_boxes(
        input_shape,
        boxes.clone(),
        original_shape,
        ratio_pad=ratio_pad,
        xywh=True,
    )


def _write_obb_txt(path: Path, ops, boxes, classes, original_shape, scores=None) -> None:
    """Write normalized four-corner OBB labels accepted by the figure generator."""
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(original_shape[0]), int(original_shape[1])
    if not boxes.shape[0]:
        path.write_text("", encoding="utf-8")
        return
    points = ops.xywhr2xyxyxyxy(boxes).reshape(-1, 4, 2).float().cpu()
    points[..., 0].div_(float(width)).clamp_(0.0, 1.0)
    points[..., 1].div_(float(height)).clamp_(0.0, 1.0)
    class_values = classes.reshape(-1).long().cpu().tolist()
    score_values = None if scores is None else scores.reshape(-1).float().cpu().tolist()
    lines = []
    for index, (class_id, polygon) in enumerate(zip(class_values, points.tolist())):
        coordinates = " ".join(f"{coordinate:.8f}" for point in polygon for coordinate in point)
        line = f"{class_id} {coordinates}"
        if score_values is not None:
            line += f" {score_values[index]:.8f}"
        lines.append(line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _relative(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    from ultralytics.utils import ops

    from myscripts.V3.runtime import build_dataset, full_loader, pad_detections, sha256_file

    output_dir = Path(args.output_dir)
    if not args.exist_ok and any((output_dir / name).exists() for name in ("manifest_all.csv", "export_audit.json")):
        raise FileExistsError(f"qualitative export already exists: {output_dir}; pass --exist-ok to overwrite")
    for name in ("images", "gt", "baseline", "ca", "refined"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    refine_bundle = load_refine_bundle(
        args.checkpoint,
        args.ca_weights,
        device_arg=args.device,
        amp=args.amp,
        imgsz=args.imgsz,
    )
    baseline_bundle = None
    try:
        baseline_bundle = load_obb_detector(
            args.baseline_weights,
            device=refine_bundle.device,
            amp=refine_bundle.use_amp,
            conf=refine_bundle.extractor.conf,
            nms_iou=refine_bundle.extractor.nms_iou,
            max_det=refine_bundle.extractor.max_det,
            expected_reg_max=16,
        )
        if baseline_bundle.extractor.nc != refine_bundle.extractor.nc:
            raise RuntimeError(
                f"class-count mismatch: Baseline nc={baseline_bundle.extractor.nc}, CA nc={refine_bundle.extractor.nc}"
            )

        dataset, _data = build_dataset(args.data, args.split, args.imgsz, args.batch, args.workers, rect=True)
        loader = full_loader(dataset, args.batch, args.workers)
        torch = refine_bundle.torch
        rows: list[dict[str, Any]] = []
        totals = {"gt": 0, "baseline": 0, "ca": 0, "refined": 0}
        identity_max_abs_box_delta = 0.0

        with torch.inference_mode():
            for batch in loader:
                _baseline_images, _bp2, _bp3, baseline_detections = baseline_bundle.extractor.infer(batch)
                images, p2, p3, ca_detections = refine_bundle.extractor.infer(batch)
                boxes, scores, classes, valid = pad_detections(ca_detections)
                with torch.autocast(
                    device_type=refine_bundle.device.type,
                    dtype=torch.float16,
                    enabled=refine_bundle.use_amp,
                ):
                    output = refine_bundle.refiner(p2, p3, boxes, scores, images.shape[2:], valid)
                residual = output["residual"].float()

                for image_index, image_path_value in enumerate(batch["im_file"]):
                    if args.max_images and len(rows) >= args.max_images:
                        break
                    image_path = Path(image_path_value)
                    scene_id = stable_export_id(image_path)
                    original_shape = batch["ori_shape"][image_index]
                    ratio_pad = batch["ratio_pad"][image_index]
                    input_shape = batch["img"].shape[2:]

                    gt_mask = batch["batch_idx"] == image_index
                    gt_boxes = batch["bboxes"][gt_mask].to(refine_bundle.device).float().clone()
                    gt_classes = batch["cls"][gt_mask].reshape(-1).to(refine_bundle.device)
                    if gt_boxes.shape[0]:
                        gt_boxes[:, :4] *= gt_boxes.new_tensor(
                            (input_shape[1], input_shape[0], input_shape[1], input_shape[0])
                        )
                    gt_boxes = _scale_rboxes(ops, gt_boxes, input_shape, original_shape, ratio_pad)

                    baseline = baseline_detections[image_index]
                    baseline_boxes = _scale_rboxes(
                        ops, baseline["bboxes"].float(), input_shape, original_shape, ratio_pad
                    )
                    ca = ca_detections[image_index]
                    ca_boxes = _scale_rboxes(ops, ca["bboxes"].float(), input_shape, original_shape, ratio_pad)

                    count = int(valid[image_index].sum().item())
                    proposal_boxes = boxes[image_index, :count].float()
                    identity_boxes = refine_bundle.refiner.apply_residual(
                        proposal_boxes, torch.zeros_like(residual[image_index, :count])
                    )
                    if count:
                        identity_max_abs_box_delta = max(
                            identity_max_abs_box_delta,
                            float((identity_boxes - proposal_boxes).abs().max().item()),
                        )
                    refined_boxes_input = refine_bundle.refiner.apply_residual(
                        proposal_boxes, residual[image_index, :count]
                    )
                    refined_boxes = _scale_rboxes(
                        ops, refined_boxes_input, input_shape, original_shape, ratio_pad
                    )

                    gt_path = output_dir / "gt" / f"{scene_id}.txt"
                    baseline_path = output_dir / "baseline" / f"{scene_id}.txt"
                    ca_path = output_dir / "ca" / f"{scene_id}.txt"
                    refined_path = output_dir / "refined" / f"{scene_id}.txt"
                    _write_obb_txt(gt_path, ops, gt_boxes, gt_classes, original_shape)
                    _write_obb_txt(
                        baseline_path,
                        ops,
                        baseline_boxes,
                        baseline["cls"],
                        original_shape,
                        baseline["conf"],
                    )
                    _write_obb_txt(ca_path, ops, ca_boxes, ca["cls"], original_shape, ca["conf"])
                    _write_obb_txt(
                        refined_path,
                        ops,
                        refined_boxes,
                        classes[image_index, :count],
                        original_shape,
                        scores[image_index, :count],
                    )

                    if args.copy_images:
                        copied_image = output_dir / "images" / f"{scene_id}{image_path.suffix.lower()}"
                        shutil.copy2(image_path, copied_image)
                        manifest_image = _relative(copied_image, output_dir)
                    else:
                        manifest_image = str(image_path)
                    row = {
                        "scene_id": scene_id,
                        "image_path": manifest_image,
                        "gt_path": _relative(gt_path, output_dir),
                        "baseline_path": _relative(baseline_path, output_dir),
                        "ca_path": _relative(ca_path, output_dir),
                        "final_path": _relative(refined_path, output_dir),
                        "roi_x1": "",
                        "roi_y1": "",
                        "roi_x2": "",
                        "roi_y2": "",
                        "note": "",
                        "original_image_path": str(image_path),
                        "image_sha256": sha256_file(image_path),
                        "original_height": int(original_shape[0]),
                        "original_width": int(original_shape[1]),
                        "input_height": int(input_shape[0]),
                        "input_width": int(input_shape[1]),
                        "gt_count": int(gt_boxes.shape[0]),
                        "baseline_count": int(baseline_boxes.shape[0]),
                        "ca_count": int(ca_boxes.shape[0]),
                        "refined_count": int(refined_boxes.shape[0]),
                    }
                    rows.append(row)
                    totals["gt"] += row["gt_count"]
                    totals["baseline"] += row["baseline_count"]
                    totals["ca"] += row["ca_count"]
                    totals["refined"] += row["refined_count"]
                if args.max_images and len(rows) >= args.max_images:
                    break

        if not rows:
            raise RuntimeError("no validation image was exported")
        if totals["ca"] != totals["refined"]:
            raise RuntimeError("CA and Refine proposal counts differ; all-proposal identity was not preserved")
        baseline_hash_after = sha256_file(baseline_bundle.weights_path)
        ca_hash_after = sha256_file(refine_bundle.ca_path)
        refine_hash_after = sha256_file(refine_bundle.checkpoint_path)
        weights_unchanged = (
            baseline_hash_after == baseline_bundle.weights_hash
            and ca_hash_after == refine_bundle.ca_hash
            and refine_hash_after == refine_bundle.checkpoint_hash
        )
        if not weights_unchanged:
            raise RuntimeError("Baseline, CA, or Refine checkpoint changed during qualitative export")
        input_shape_counts: dict[str, int] = {}
        for row in rows:
            shape = f"{row['input_height']}x{row['input_width']}"
            input_shape_counts[shape] = input_shape_counts.get(shape, 0) + 1
        audit = {
            "tool": "export_qualitative_v311",
            "data": args.data,
            "split": args.split,
            "test_used": False,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "amp": refine_bundle.use_amp,
            "device": str(refine_bundle.device),
            "workers": args.workers,
            "image_count": len(rows),
            "input_shape_counts": input_shape_counts,
            "copied_images": args.copy_images,
            "baseline_weights": str(baseline_bundle.weights_path),
            "baseline_sha256": baseline_bundle.weights_hash,
            "baseline_reg_max": baseline_bundle.reg_max,
            "ca_weights": str(refine_bundle.ca_path),
            "ca_sha256": refine_bundle.ca_hash,
            "ca_reg_max": 32,
            "refine_checkpoint": str(refine_bundle.checkpoint_path),
            "refine_sha256": refine_bundle.checkpoint_hash,
            "refine_epoch": refine_bundle.checkpoint.get("epoch"),
            "confidence": refine_bundle.extractor.conf,
            "nms_iou": refine_bundle.extractor.nms_iou,
            "max_det": refine_bundle.extractor.max_det,
            "proposal_policy": "all post-NMS proposals",
            "rerun_nms": False,
            "score_changed": False,
            "class_changed": False,
            "identity_max_abs_box_delta": identity_max_abs_box_delta,
            "identity_pass": identity_max_abs_box_delta == 0.0,
            "weights_unchanged": weights_unchanged,
            "totals": totals,
            "manifest": str(output_dir / "manifest_all.csv"),
        }
        if not audit["identity_pass"]:
            raise RuntimeError(f"zero-residual identity failed: max abs box delta={identity_max_abs_box_delta}")
        write_csv(output_dir / "manifest_all.csv", rows)
        write_json(output_dir / "export_audit.json", audit)
        print(output_dir / "export_audit.json")
    finally:
        if baseline_bundle is not None:
            baseline_bundle.close()
        refine_bundle.close()


if __name__ == "__main__":
    main()
