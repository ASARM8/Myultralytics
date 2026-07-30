"""Evaluate coarse-only and Refine inference variants over one or more checkpoints.

The default ``curve`` profile preserves the original paper-curve behavior. Diagnostic
profiles add identity, residual-scale, and gate controls. Unless ``--shared-model`` is
specified, every validation variant reloads the checkpoint independently so that a
stateful validation side effect cannot masquerade as a Refine effect.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import pandas as pd


@dataclass(frozen=True)
class EvalVariant:
    """One Refine inference configuration."""

    key: str
    method: str
    coarse_only: bool
    alpha: float = 1.0
    gate_mode: str = "current"


def parse_weight_spec(value: str) -> tuple[int, Path]:
    """Parse ``EPOCH=PATH``."""
    if "=" not in value:
        raise ValueError(f"--weights 必须使用 EPOCH=PATH: {value}")
    epoch, path = value.split("=", 1)
    return int(epoch), Path(path)


def discover_weights(pattern: str, epoch_regex: str, epoch_offset: int) -> list[tuple[int, Path]]:
    """Discover checkpoints and convert filename indices to plotted epoch numbers."""
    regex = re.compile(epoch_regex)
    path = Path(pattern)
    paths = sorted(Path().glob(pattern)) if not path.is_absolute() else sorted(path.parent.glob(path.name))
    discovered = []
    for path in paths:
        match = regex.fullmatch(path.stem)
        if not match:
            print(f"[跳过] 权重文件名不完全匹配 --epoch-regex: {path}")
            continue
        discovered.append((int(match.group(1)) + epoch_offset, path))
    return discovered


def _diagnostic_refine_gate(self, dbox: torch.Tensor) -> torch.Tensor:
    """Build a runtime-selectable Refine gate for diagnostic validation."""
    wh = dbox[:, 2:4, :]
    short_side = wh.amin(dim=1, keepdim=True)
    long_side = wh.amax(dim=1, keepdim=True)
    aspect_ratio = long_side / short_side.clamp_min(1e-6)
    ar_gate = aspect_ratio > float(getattr(self, "refine_select_ar", 30.0))
    short_gate = short_side < float(getattr(self, "refine_select_ws", 16.0))
    gate_mode = getattr(self, "_diagnostic_refine_gate_mode", "current")
    if gate_mode == "current":
        return ar_gate | short_gate
    if gate_mode == "ar-only":
        return ar_gate
    if gate_mode == "short-only":
        return short_gate
    if gate_mode == "and":
        return ar_gate & short_gate
    if gate_mode == "all":
        return ar_gate | ~ar_gate
    if gate_mode == "none":
        return ar_gate & ~ar_gate
    raise ValueError(f"未知 Refine gate mode: {gate_mode}")


def _diagnostic_apply_wh_refine(
    self,
    dbox: torch.Tensor,
    refine: torch.Tensor,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply Refine with a runtime residual scale while preserving the production formula at alpha=1."""
    clamp_value = float(getattr(self, "refine_clamp", 1.0))
    alpha = float(getattr(self, "_diagnostic_refine_alpha", 1.0))
    gate_mode = getattr(self, "_diagnostic_refine_gate_mode", "current")
    if alpha == 0.0 or gate_mode == "none":
        return dbox.clone()
    if getattr(self, "refine_version", 1) == 2:
        delta = self.bound_refine(refine) * alpha
        return self._apply_refine_delta(dbox, delta, gate)
    dw = refine[:, 0:1, :].clamp(-clamp_value, clamp_value) * alpha
    dh = refine[:, 1:2, :].clamp(-clamp_value, clamp_value) * alpha
    if gate is not None:
        gate = gate.to(dtype=dw.dtype, device=dw.device)
        dw = dw * gate
        dh = dh * gate
    refined = dbox.clone()
    refined[:, 2:3, :] = dbox[:, 2:3, :] * dw.exp()
    refined[:, 3:4, :] = dbox[:, 3:4, :] * dh.exp()
    return refined


def configure_refine_variant(model, variant: EvalVariant) -> int:
    """Configure all Refine heads for one diagnostic variant."""
    changed = 0
    for module in model.model.modules():
        if not hasattr(module, "disable_refine_inference"):
            continue
        module.disable_refine_inference = bool(variant.coarse_only)
        module._diagnostic_refine_alpha = float(variant.alpha)
        module._diagnostic_refine_gate_mode = variant.gate_mode
        if not variant.coarse_only:
            module._build_refine_gate = MethodType(_diagnostic_refine_gate, module)
            module._apply_wh_refine = MethodType(_diagnostic_apply_wh_refine, module)
        changed += 1
    if changed == 0:
        raise RuntimeError("权重中未找到带 disable_refine_inference 属性的 Refine 模块")
    return changed


def parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated list of finite alpha values in [0, 1]."""
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("列表不能为空")
    if any(not 0.0 <= item <= 1.0 for item in values):
        raise argparse.ArgumentTypeError("alpha 必须位于 [0, 1]")
    return values


def parse_gate_list(value: str) -> list[str]:
    """Parse and validate a comma-separated gate-mode list."""
    allowed = {"current", "ar-only", "short-only", "and", "all", "none"}
    values = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in values if item not in allowed]
    if not values or invalid:
        raise argparse.ArgumentTypeError(f"gate mode 必须来自 {sorted(allowed)}，收到: {invalid or value}")
    return values


def unique_variants(variants: list[EvalVariant]) -> list[EvalVariant]:
    """Remove duplicate variants while preserving order."""
    result = []
    seen = set()
    for variant in variants:
        identity = (variant.coarse_only, variant.alpha, variant.gate_mode)
        if identity not in seen:
            result.append(variant)
            seen.add(identity)
    return result


def build_variants(args: argparse.Namespace) -> list[EvalVariant]:
    """Build the requested validation profile."""
    coarse = EvalVariant("coarse", "CA+Refine (coarse-only)", True)
    normal = EvalVariant("normal", "CA+Refine (normal refine)", False)
    gate_off = EvalVariant("gate_off", "CA+Refine (gate off identity)", False, gate_mode="none")
    zero_delta = EvalVariant("zero_delta", "CA+Refine (zero residual identity)", False, alpha=0.0)

    if args.profile == "curve":
        return [coarse, normal] if args.order == "coarse-first" else [normal, coarse]
    if args.profile == "identity":
        return [coarse, gate_off, zero_delta, normal]
    if args.profile == "alpha":
        variants = [coarse]
        variants.extend(
            EvalVariant(
                f"alpha_{alpha:g}",
                "CA+Refine (normal refine)" if alpha == 1.0 else f"CA+Refine (alpha={alpha:g})",
                False,
                alpha=alpha,
            )
            for alpha in args.alphas
        )
        return unique_variants(variants)
    if args.profile == "gate":
        variants = [coarse]
        variants.extend(
            EvalVariant(
                f"gate_{gate}_alpha_{args.gate_alpha:g}",
                f"CA+Refine (gate={gate}, alpha={args.gate_alpha:g})",
                False,
                alpha=args.gate_alpha,
                gate_mode=gate,
            )
            for gate in args.gate_modes
        )
        return unique_variants(variants)

    variants = [coarse, gate_off, zero_delta]
    variants.extend(
        EvalVariant(
            f"gate_{gate}_alpha_{alpha:g}",
            (
                "CA+Refine (normal refine)"
                if gate == "current" and alpha == 1.0
                else f"CA+Refine (gate={gate}, alpha={alpha:g})"
            ),
            False,
            alpha=alpha,
            gate_mode=gate,
        )
        for gate in args.gate_modes
        for alpha in args.alphas
    )
    return unique_variants(variants)


def verify_checkpoint_epoch(model, plotted_epoch: int, path: Path) -> None:
    """Cross-check a checkpoint's stored one-based epoch when metadata is available."""
    checkpoint = getattr(model, "ckpt", None)
    stored_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    if stored_epoch is None or int(stored_epoch) < 0:
        print(f"[提示] {path.name} 未保留可用 epoch 元数据，使用命令行 epoch={plotted_epoch}")
        return
    metadata_epoch = int(stored_epoch) + 1
    if metadata_epoch != plotted_epoch:
        raise ValueError(
            f"checkpoint epoch 不一致: {path} 内部对应 epoch={metadata_epoch}，"
            f"但命令行/文件名解析得到 epoch={plotted_epoch}"
        )


def metrics_row(
    epoch: int,
    weights: Path,
    data: str,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    split: str,
    profile: str,
    variant: EvalVariant,
    fresh_load: bool,
    results,
) -> dict:
    """Convert Ultralytics validation output to the tidy diagnostic schema."""
    values = results.results_dict
    metric = results.box
    all_ap = metric.all_ap
    threshold_ap = {
        f"ap{iou}": float(all_ap[:, index].mean()) if len(all_ap) else float("nan")
        for index, iou in enumerate(range(50, 100, 5))
    }
    return {
        "epoch": epoch,
        "method": variant.method,
        "weights": str(weights.resolve()),
        "data": data,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": workers,
        "split": split,
        "profile": profile,
        "variant": variant.key,
        "refine_alpha": variant.alpha,
        "refine_gate": variant.gate_mode,
        "fresh_load": fresh_load,
        "precision": float(values["metrics/precision(B)"]),
        "recall": float(values["metrics/recall(B)"]),
        "map50": float(values["metrics/mAP50(B)"]),
        "map50_95": float(values["metrics/mAP50-95(B)"]),
        "ap75": float(metric.map75),
        "ap90": threshold_ap["ap90"],
        "ap55": threshold_ap["ap55"],
        "ap60": threshold_ap["ap60"],
        "ap65": threshold_ap["ap65"],
        "ap70": threshold_ap["ap70"],
        "ap80": threshold_ap["ap80"],
        "ap85": threshold_ap["ap85"],
        "ap95": threshold_ap["ap95"],
    }


def main():
    """Run the requested checkpoint validation profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", action="append", default=[], metavar="EPOCH=PATH")
    parser.add_argument("--weights-glob", help="例如 work-dirs/exp/weights/epoch*.pt")
    parser.add_argument("--epoch-regex", default=r"epoch(\d+)")
    parser.add_argument(
        "--glob-epoch-offset",
        type=int,
        default=1,
        help="文件名 epoch 索引到图中 epoch 的偏移；epochN.pt 默认对应 results.csv 中的 N+1",
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--project", default="work-dirs/paper_refine_ab_curve")
    parser.add_argument(
        "--profile",
        choices=("curve", "identity", "alpha", "gate", "full"),
        default="curve",
        help="curve=原始两模式曲线；identity=恒等性检查；alpha=残差尺度；gate=门控拆分；full=全部诊断",
    )
    parser.add_argument("--order", choices=("coarse-first", "normal-first"), default="coarse-first")
    parser.add_argument(
        "--alphas",
        type=parse_float_list,
        default=parse_float_list("0,0.05,0.1,0.2,0.5,1"),
        help="alpha/full profile 的逗号分隔残差系数",
    )
    parser.add_argument(
        "--gate-modes",
        type=parse_gate_list,
        default=parse_gate_list("current,ar-only,short-only,and"),
        help="gate/full profile 的逗号分隔门控模式",
    )
    parser.add_argument("--gate-alpha", type=float, default=1.0, help="gate profile 使用的残差系数")
    parser.add_argument(
        "--identity-tolerance",
        type=float,
        default=5e-4,
        help="identity profile 与 coarse 的最大允许指标差值",
    )
    parser.add_argument(
        "--shared-model",
        action="store_true",
        help="各模式复用同一模型实例；默认每个模式重新加载权重以排除状态污染",
    )
    args = parser.parse_args()

    if args.imgsz != 640:
        parser.error("创新点一图表固定使用 imgsz=640")
    if not 0.0 <= args.gate_alpha <= 1.0:
        parser.error("--gate-alpha 必须位于 [0, 1]")
    if args.identity_tolerance < 0:
        parser.error("--identity-tolerance 不能为负数")

    checkpoints = [parse_weight_spec(value) for value in args.weights]
    if args.weights_glob:
        checkpoints.extend(discover_weights(args.weights_glob, args.epoch_regex, args.glob_epoch_offset))
    if not checkpoints:
        raise ValueError("未提供任何 checkpoint")
    by_epoch = {}
    for epoch, path in checkpoints:
        if epoch in by_epoch and by_epoch[epoch] != path:
            raise ValueError(f"同一 epoch 重复指定了不同 checkpoint: epoch={epoch}, {by_epoch[epoch]}, {path}")
        by_epoch[epoch] = path
    checkpoints = sorted(by_epoch.items())

    from ultralytics import YOLO

    rows = []
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    variants = build_variants(args)
    if args.profile == "full":
        print(f"[提示] full profile 每个 checkpoint 将运行 {len(variants)} 次完整验证，建议只传一个代表性权重。")

    for epoch, path in checkpoints:
        if not path.exists():
            raise FileNotFoundError(path)
        common = {
            "data": args.data,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "project": args.project,
            "plots": False,
            "save_json": False,
            "verbose": False,
            "split": args.split,
        }
        shared_model = None
        if args.shared_model:
            shared_model = YOLO(str(path))
            verify_checkpoint_epoch(shared_model, epoch, path)

        epoch_rows = []
        for variant in variants:
            model = shared_model if shared_model is not None else YOLO(str(path))
            if shared_model is None:
                verify_checkpoint_epoch(model, epoch, path)
            configure_refine_variant(model, variant)
            results = model.val(name=f"epoch{epoch:04d}_{variant.key}", exist_ok=True, **common)
            row = metrics_row(
                epoch=epoch,
                weights=path,
                data=args.data,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                split=args.split,
                profile=args.profile,
                variant=variant,
                fresh_load=not args.shared_model,
                results=results,
            )
            rows.append(row)
            epoch_rows.append(row)
            pd.DataFrame(rows).to_csv(args.output_csv, index=False, encoding="utf-8-sig")
            print(
                f"epoch={epoch} variant={variant.key}: mAP50-95={row['map50_95']:.6f}, "
                f"AP75={row['ap75']:.6f}, AP90={row['ap90']:.6f}"
            )

        coarse_rows = [row for row in epoch_rows if row["variant"] == "coarse"]
        if coarse_rows:
            coarse_map = coarse_rows[0]["map50_95"]
            for row in epoch_rows:
                if row["variant"] != "coarse":
                    print(f"  Δ({row['variant']}-coarse)={row['map50_95'] - coarse_map:+.6f} [mAP50-95]")
            if args.profile == "identity":
                coarse_row = coarse_rows[0]
                for row in epoch_rows:
                    if row["variant"] not in {"gate_off", "zero_delta"}:
                        continue
                    differences = {
                        metric: abs(row[metric] - coarse_row[metric])
                        for metric in (
                            "precision",
                            "recall",
                            "map50",
                            "map50_95",
                            "ap55",
                            "ap60",
                            "ap65",
                            "ap70",
                            "ap75",
                            "ap80",
                            "ap85",
                            "ap90",
                            "ap95",
                        )
                    }
                    maximum = max(differences.values())
                    status = "PASS" if maximum <= args.identity_tolerance else "FAIL"
                    print(
                        f"  identity {row['variant']}: {status}, max_abs_diff={maximum:.6g}, "
                        f"tolerance={args.identity_tolerance:.6g}"
                    )

    print(args.output_csv)


if __name__ == "__main__":
    main()
