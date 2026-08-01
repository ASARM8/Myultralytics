"""Audit Refine checkpoint drift without running dataset inference.

The report separates trainable ``cv5`` parameters, ``cv5`` buffers, frozen CA
parameters, and frozen CA buffers. This prevents BatchNorm running statistics from
being mistaken for optimizer updates.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
from pathlib import Path


def discover_paths(explicit: list[Path], patterns: list[str]) -> list[Path]:
    """Expand explicit paths and glob patterns with stable de-duplication."""
    candidates = [*explicit]
    for pattern in patterns:
        candidates.extend(Path(item) for item in glob.glob(pattern))
    result = []
    seen = set()
    for path in candidates:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            if not resolved.exists():
                raise FileNotFoundError(resolved)
            result.append(resolved)
            seen.add(resolved)
    return result


def tensor_digest(state_dict, predicate, torch) -> tuple[str, list[str]]:
    """Hash selected tensors by name, shape, dtype and exact bytes."""
    digest = hashlib.sha256()
    keys = [key for key in sorted(state_dict) if predicate(key)]
    for key in keys:
        value = state_dict[key].detach().cpu().contiguous().reshape(-1)
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(state_dict[key].shape)).encode("ascii"))
        digest.update(str(state_dict[key].dtype).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest(), keys


def find_refine_head(core_model):
    """Return the unique unfused Refine head."""
    heads = [
        module
        for module in core_model.modules()
        if hasattr(module, "disable_refine_inference") and hasattr(module, "cv5")
    ]
    if len(heads) != 1 or heads[0].cv5 is None:
        raise RuntimeError(f"需要且仅允许一个未融合 Refine Head，实际找到 {len(heads)} 个")
    return heads[0]


def inspect_checkpoint(path: Path, torch, yolo_class) -> dict:
    """Collect exact hashes and compact cv5 magnitude statistics."""
    yolo = yolo_class(str(path))
    core_model = yolo.model
    head = find_refine_head(core_model)
    state = core_model.state_dict()

    def is_refine(key: str) -> bool:
        return ".cv5." in key or ".one2one_cv5." in key

    parameter_names = {name for name, _ in core_model.named_parameters()}
    buffer_names = {name for name, _ in core_model.named_buffers()}
    cv5_parameter_digest, cv5_parameter_keys = tensor_digest(
        state,
        lambda key: is_refine(key) and key in parameter_names,
        torch,
    )
    cv5_buffer_digest, cv5_buffer_keys = tensor_digest(
        state,
        lambda key: is_refine(key) and key in buffer_names,
        torch,
    )
    shared_parameter_digest, shared_parameter_keys = tensor_digest(
        state,
        lambda key: not is_refine(key) and key in parameter_names,
        torch,
    )
    shared_buffer_digest, shared_buffer_keys = tensor_digest(
        state,
        lambda key: (
            not is_refine(key)
            and key in buffer_names
            and not key.endswith("._refine_v2_marker")
        ),
        torch,
    )
    if not cv5_parameter_keys:
        raise RuntimeError(f"checkpoint 未找到 cv5 可训练参数: {path}")
    cv5_values = torch.cat(
        [state[key].detach().float().cpu().reshape(-1) for key in cv5_parameter_keys]
    )
    checkpoint = getattr(yolo, "ckpt", {})
    stored_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    plotted_epoch = int(stored_epoch) + 1 if stored_epoch is not None and int(stored_epoch) >= 0 else None
    return {
        "checkpoint": str(path),
        "file": path.name,
        "epoch": plotted_epoch,
        "refine_version": int(getattr(head, "refine_version", 1)),
        "refine_experiment": str(getattr(head, "refine_experiment", "legacy")),
        "refine_delta_max": getattr(head, "refine_delta_max", None),
        "refine_target_limit": getattr(head, "refine_target_limit", None),
        "cv5_parameter_tensor_count": len(cv5_parameter_keys),
        "cv5_parameter_count": int(cv5_values.numel()),
        "cv5_mean": float(cv5_values.mean().item()),
        "cv5_std": float(cv5_values.std(unbiased=False).item()),
        "cv5_max_abs": float(cv5_values.abs().max().item()),
        "cv5_parameter_sha256": cv5_parameter_digest,
        "cv5_buffer_tensor_count": len(cv5_buffer_keys),
        "cv5_buffer_sha256": cv5_buffer_digest,
        "shared_parameter_tensor_count": len(shared_parameter_keys),
        "shared_parameter_sha256": shared_parameter_digest,
        "shared_buffer_tensor_count": len(shared_buffer_keys),
        "shared_buffer_sha256": shared_buffer_digest,
    }


def main() -> None:
    """Write one audit row per checkpoint and print invariant failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", action="append", type=Path, default=[], help="可重复指定 checkpoint")
    parser.add_argument("--weights-glob", action="append", default=[], help="可重复指定 glob，例如 weights/epoch*.pt")
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    paths = discover_paths(args.weights, args.weights_glob)
    if len(paths) < 2:
        parser.error("至少需要两个 checkpoint 才能判断参数漂移")

    import torch

    from ultralytics import YOLO

    rows = [inspect_checkpoint(path, torch, YOLO) for path in paths]
    rows.sort(key=lambda row: (row["epoch"] is None, row["epoch"] or 0, row["checkpoint"]))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    shared_parameter_hashes = {row["shared_parameter_sha256"] for row in rows}
    shared_buffer_hashes = {row["shared_buffer_sha256"] for row in rows}
    cv5_parameter_hashes = {row["cv5_parameter_sha256"] for row in rows}
    cv5_buffer_hashes = {row["cv5_buffer_sha256"] for row in rows}
    print(f"已检查 {len(rows)} 个 checkpoint，结果: {args.output_csv}")
    print(
        f"shared parameter hashes={len(shared_parameter_hashes)} "
        f"({'PASS' if len(shared_parameter_hashes) == 1 else 'FAIL'})"
    )
    print(
        f"shared buffer hashes={len(shared_buffer_hashes)} "
        f"({'PASS' if len(shared_buffer_hashes) == 1 else 'FAIL'})"
    )
    print(
        f"cv5 parameter hashes={len(cv5_parameter_hashes)} "
        f"({'PASS' if len(cv5_parameter_hashes) > 1 else 'FAIL'})"
    )
    print(f"cv5 buffer hashes={len(cv5_buffer_hashes)} (diagnostic only)")
    if len(shared_parameter_hashes) != 1:
        print("[FAIL] 冻结的 CA 可训练参数在 checkpoint 间发生变化。")
    if len(shared_buffer_hashes) != 1:
        print("[FAIL] 冻结的 CA 缓冲区在 checkpoint 间发生变化，重点检查 BatchNorm eval 状态。")
    if len(cv5_parameter_hashes) == 1:
        print("[FAIL] 所有 checkpoint 的 cv5 可训练参数完全相同，需检查优化器、保存或加载链。")
    elif len(cv5_parameter_hashes) < len(rows):
        print("[WARN] 部分 checkpoint 的 cv5 参数哈希重复，请结合 epoch 行定位。")
    if (
        len(shared_parameter_hashes) != 1
        or len(shared_buffer_hashes) != 1
        or len(cv5_parameter_hashes) == 1
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
