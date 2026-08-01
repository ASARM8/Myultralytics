"""Diagnose the OBB Refine branch on the 640-pixel validation/test split.

This script does not replace full mAP validation. It inspects the mechanism behind
the result: predicted-vs-GT gate mismatch, raw residual magnitude and saturation,
per-FPN activation, matched-positive IoU changes, and optional gradient-clipping
coupling. Use ``collect_refine_ab_curve.py`` for full-dataset metric controls.

Example:
    python myscripts/refine_diag.py \
        --weights /root/autodl-tmp/work-dirs/exp/weights/best.pt \
        --data /root/autodl-tmp/datasets/TTPLA-640-811/dataset.yaml \
        --imgsz 640 --split val --batch 8 --device 0 --workers 8 \
        --max-batches 50 \
        --output-dir /root/autodl-tmp/paper_exports/refine_diag_best
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

GATE_MODES = ("current", "ar-only", "short-only", "and", "all", "none")
REFINE_RUNTIME_FIELDS = (
    "refine_experiment",
    "refine_delta_max",
    "refine_target_limit",
    "refine_smooth_l1_beta",
    "refine_identity_gain",
    "refine_feature_detach",
)


def parse_float_list(value: str, *, minimum: float | None = None, maximum: float | None = None) -> list[float]:
    """Parse a comma-separated float list with optional bounds."""
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("列表不能为空")
    if minimum is not None and any(item < minimum for item in result):
        raise argparse.ArgumentTypeError(f"列表值不能小于 {minimum}")
    if maximum is not None and any(item > maximum for item in result):
        raise argparse.ArgumentTypeError(f"列表值不能大于 {maximum}")
    return result


def parse_alphas(value: str) -> list[float]:
    """Parse residual scales in [0, 1]."""
    return parse_float_list(value, minimum=0.0, maximum=1.0)


def parse_thresholds(value: str) -> list[float]:
    """Parse non-negative confidence thresholds."""
    return parse_float_list(value, minimum=0.0, maximum=1.0)


def parse_gate_modes(value: str) -> list[str]:
    """Parse gate modes."""
    result = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in result if item not in GATE_MODES]
    if not result or invalid:
        raise argparse.ArgumentTypeError(f"gate mode 必须来自 {GATE_MODES}，收到: {invalid or value}")
    return result


def find_refine_head(core_model: torch.nn.Module) -> torch.nn.Module:
    """Find the unique OBBRefine-like head."""
    heads = [
        module
        for module in core_model.modules()
        if hasattr(module, "disable_refine_inference") and hasattr(module, "cv5")
    ]
    if not heads:
        raise RuntimeError("权重中未找到 OBBRefine 检测头；请确认使用 CA+Refine checkpoint")
    if len(heads) > 1:
        raise RuntimeError(f"检测到 {len(heads)} 个 Refine head，脚本无法确定目标模块")
    head = heads[0]
    if head.cv5 is None:
        raise RuntimeError("Refine head 的 cv5 已被移除，无法采集原始残差")
    return head


def read_refine_runtime_args(refine_head: torch.nn.Module) -> dict[str, Any]:
    """Capture checkpoint-persisted Refine attributes before model args are normalized."""
    missing = [name for name in REFINE_RUNTIME_FIELDS if not hasattr(refine_head, name)]
    if missing:
        raise RuntimeError(f"Refine checkpoint 缺少运行时属性: {missing}")
    return {name: getattr(refine_head, name) for name in REFINE_RUNTIME_FIELDS}


def assert_refine_runtime_args(refine_head: torch.nn.Module, expected: dict[str, Any]) -> None:
    """Fail if criterion initialization silently replaces checkpoint Refine semantics."""
    mismatches = []
    for name in REFINE_RUNTIME_FIELDS:
        actual = getattr(refine_head, name, None)
        wanted = expected[name]
        equal = (
            math.isclose(float(actual), float(wanted), rel_tol=0.0, abs_tol=1e-12)
            if isinstance(wanted, float)
            else actual == wanted
        )
        if not equal:
            mismatches.append(f"{name}: checkpoint={wanted!r}, runtime={actual!r}")
    if mismatches:
        raise RuntimeError("Refine 运行时参数被覆盖:\n  " + "\n  ".join(mismatches))


def configure_model_args(
    core_model: torch.nn.Module,
    args: argparse.Namespace,
    refine_runtime_args: dict[str, Any],
) -> Any:
    """Normalize model args while restoring custom Refine values stripped by checkpoint loading."""
    stored_args = getattr(core_model, "args", {})
    overrides = dict(stored_args) if isinstance(stored_args, dict) else vars(stored_args)
    overrides = {key: value for key, value in overrides.items() if key in DEFAULT_CFG_KEYS}
    overrides.update(refine_runtime_args)
    overrides.update(
        {
            "task": "obb",
            "data": args.data,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "rect": True,
            "cache": False,
            "aux_geo_ar": args.ar_threshold,
            "aux_geo_ws": args.short_threshold,
        }
    )
    cfg = get_cfg(DEFAULT_CFG, overrides=overrides)
    core_model.args = cfg
    return cfg


def build_split_loader(
    cfg: Any,
    data_dict: dict[str, Any],
    split: str,
    batch: int,
    workers: int,
    stride: int,
):
    """Build a deterministic validation-style loader for ``val`` or ``test``."""
    image_path = data_dict.get(split)
    if not image_path:
        raise ValueError(f"dataset.yaml 未定义 split={split}")
    dataset = build_yolo_dataset(
        cfg,
        image_path,
        batch,
        data_dict,
        mode="val",
        rect=True,
        stride=stride,
    )
    return build_dataloader(
        dataset,
        batch,
        workers,
        shuffle=False,
        rank=-1,
        drop_last=False,
        pin_memory=True,
    )


def prepare_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor fields to the selected device and normalize images."""
    prepared = {}
    for key, value in batch.items():
        prepared[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    prepared["img"] = prepared["img"].float() / 255.0
    return prepared


def extract_raw_predictions(outputs: Any) -> dict[str, torch.Tensor]:
    """Extract the one-to-many raw prediction dictionary."""
    raw = outputs[1] if isinstance(outputs, (tuple, list)) else outputs
    if isinstance(raw, dict) and "one2many" in raw:
        raw = raw["one2many"]
    if not isinstance(raw, dict):
        raise TypeError(f"无法从模型输出中取得 raw prediction dict，实际类型={type(raw)}")
    required = {"boxes", "scores", "angle", "refine", "feats"}
    missing = required.difference(raw)
    if missing:
        raise KeyError(f"raw predictions 缺少字段: {sorted(missing)}")
    return raw


def build_assignments(
    criterion,
    raw: dict[str, torch.Tensor],
    batch: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Reproduce the coarse assignment path used by the OBB loss."""
    pred_distri = raw["boxes"].permute(0, 2, 1).contiguous()
    pred_scores = raw["scores"].permute(0, 2, 1).contiguous()
    pred_angle = raw["angle"].permute(0, 2, 1).contiguous()
    anchor_points, stride_tensor = make_anchors(raw["feats"], criterion.stride, 0.5)
    coarse_grid = criterion.bbox_decode(anchor_points, pred_distri, pred_angle)
    coarse_px = coarse_grid.clone()
    coarse_px[..., :4] *= stride_tensor

    batch_size = pred_angle.shape[0]
    dtype = pred_scores.dtype
    image_size = torch.tensor(
        raw["feats"][0].shape[2:],
        device=criterion.device,
        dtype=dtype,
    ) * criterion.stride[0]
    batch_index = batch["batch_idx"].view(-1, 1)
    targets = torch.cat((batch_index, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
    rw = targets[:, 4] * float(image_size[1])
    rh = targets[:, 5] * float(image_size[0])
    targets = targets[(rw >= 2) & (rh >= 2)]
    targets = criterion.preprocess(
        targets.to(criterion.device),
        batch_size,
        scale_tensor=image_size[[1, 0, 1, 0]],
    )
    gt_labels, gt_bboxes = targets.split((1, 5), 2)
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

    criterion.assigner._stride_tensor = stride_tensor
    _, target_bboxes, target_scores, fg_mask, target_gt_idx = criterion.assigner(
        pred_scores.detach().sigmoid(),
        coarse_px.detach().type(gt_bboxes.dtype),
        anchor_points * stride_tensor,
        gt_labels,
        gt_bboxes,
        mask_gt,
    )
    return {
        "coarse_grid": coarse_grid,
        "coarse_px": coarse_px,
        "pred_scores": pred_scores,
        "pred_refine": raw["refine"].permute(0, 2, 1).contiguous(),
        "target_bboxes": target_bboxes,
        "target_scores": target_scores,
        "target_gt_idx": target_gt_idx,
        "fg_mask": fg_mask,
        "stride_tensor": stride_tensor,
    }


def build_predicted_gates(
    coarse_px: torch.Tensor,
    ar_threshold: float,
    short_threshold: float,
) -> dict[str, torch.Tensor]:
    """Build all inference-side gate variants from coarse predicted geometry."""
    short_side = coarse_px[..., 2:4].amin(dim=-1)
    long_side = coarse_px[..., 2:4].amax(dim=-1)
    aspect_ratio = long_side / short_side.clamp_min(1e-6)
    ar_gate = aspect_ratio > ar_threshold
    short_gate = short_side < short_threshold
    return {
        "current": ar_gate | short_gate,
        "ar-only": ar_gate,
        "short-only": short_gate,
        "and": ar_gate & short_gate,
        "all": torch.ones_like(ar_gate),
        "none": torch.zeros_like(ar_gate),
    }


def build_gt_gate(
    target_fg: torch.Tensor,
    ar_threshold: float,
    short_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the training-side GT gate and return GT short side/aspect ratio."""
    short_side = target_fg[:, 2:4].amin(dim=-1)
    long_side = target_fg[:, 2:4].amax(dim=-1)
    aspect_ratio = long_side / short_side.clamp_min(1e-6)
    gate = (aspect_ratio > ar_threshold) | (short_side < short_threshold)
    return gate, short_side, aspect_ratio


def apply_refine(
    coarse_bboxes: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    alpha: float,
    clamp_value: float,
    refine_head: torch.nn.Module,
) -> torch.Tensor:
    """Apply the active legacy/V2 production residual formula to selected boxes."""
    refined = coarse_bboxes.clone()
    delta = effective_residual(residual, refine_head, clamp_value)
    delta_wh = residual_to_wh_delta(coarse_bboxes, delta, refine_head)
    delta_wh = delta_wh * float(alpha) * gate.to(delta_wh.dtype).unsqueeze(-1)
    refined[:, 2:4] *= torch.exp(delta_wh)
    return refined


def apply_short_long_target(
    coarse_bboxes: torch.Tensor,
    target_delta: torch.Tensor,
    gate: torch.Tensor,
    refine_head: torch.nn.Module,
) -> torch.Tensor:
    """Apply clipped direct targets to quantify the attainable scale-only oracle."""
    refined = coarse_bboxes.clone()
    delta_wh = residual_to_wh_delta(coarse_bboxes, target_delta, refine_head)
    delta_wh = delta_wh * gate.to(delta_wh.dtype).unsqueeze(-1)
    refined[:, 2:4] *= torch.exp(delta_wh)
    return refined


def effective_residual(
    residual: torch.Tensor,
    refine_head: torch.nn.Module,
    clamp_value: float,
) -> torch.Tensor:
    """Return the residual actually applied before alpha and gate."""
    if getattr(refine_head, "refine_version", 1) == 2:
        return refine_head.bound_refine(residual)
    return residual[..., :2].clamp(-clamp_value, clamp_value)


def residual_to_wh_delta(
    coarse_bboxes: torch.Tensor,
    delta: torch.Tensor,
    refine_head: torch.nn.Module,
) -> torch.Tensor:
    """Map active residual channels to width/height order."""
    experiment = getattr(refine_head, "refine_experiment", "legacy")
    if getattr(refine_head, "refine_version", 1) == 1 or experiment == "bounded_wh":
        return delta[..., :2]
    short_is_width = coarse_bboxes[..., 2] <= coarse_bboxes[..., 3]
    delta_w = torch.where(short_is_width, delta[..., 0], delta[..., 1])
    delta_h = torch.where(short_is_width, delta[..., 1], delta[..., 0])
    return torch.stack((delta_w, delta_h), dim=-1)


def extend_values(
    store: dict[tuple[str, str], list[np.ndarray]],
    scope: str,
    variable: str,
    values: torch.Tensor,
) -> None:
    """Append finite flattened values to a distribution store."""
    if values.numel() == 0:
        return
    array = values.detach().float().reshape(-1).cpu().numpy()
    array = array[np.isfinite(array)]
    if array.size:
        store[(scope, variable)].append(array)


def extend_alignment(
    store: dict[tuple[str, str], dict[str, list[np.ndarray]]],
    scope: str,
    channel: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    """Append paired effective residual predictions and direct targets."""
    if prediction.numel() == 0:
        return
    pred_array = prediction.detach().float().reshape(-1).cpu().numpy()
    target_array = target.detach().float().reshape(-1).cpu().numpy()
    finite = np.isfinite(pred_array) & np.isfinite(target_array)
    if finite.any():
        store[(scope, channel)]["prediction"].append(pred_array[finite])
        store[(scope, channel)]["target"].append(target_array[finite])


def add_gate_count(
    counts: dict[tuple[str, str], dict[str, int]],
    scope: str,
    gate_name: str,
    gate: torch.Tensor,
    eligible: torch.Tensor | None = None,
) -> None:
    """Accumulate active and eligible gate counts."""
    selected = gate if eligible is None else gate[eligible]
    entry = counts[(scope, gate_name)]
    entry["active"] += int(selected.sum().item())
    entry["total"] += int(selected.numel())


def add_iou_values(
    store: dict[tuple[str, float, str], dict[str, list[np.ndarray]]],
    gate_name: str,
    alpha: float,
    subset: str,
    coarse_iou: torch.Tensor,
    refined_iou: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> None:
    """Accumulate paired IoU arrays for one configuration/subgroup."""
    if mask is not None:
        coarse_iou = coarse_iou[mask]
        refined_iou = refined_iou[mask]
    if coarse_iou.numel() == 0:
        return
    key = (gate_name, float(alpha), subset)
    store[key]["coarse"].append(coarse_iou.detach().float().reshape(-1).cpu().numpy())
    store[key]["refined"].append(refined_iou.detach().float().reshape(-1).cpu().numpy())


def fpn_level_ids(feats: list[torch.Tensor], batch_size: int, device: torch.device) -> tuple[torch.Tensor, list[str]]:
    """Return flattened P-level indices aligned with concatenated head predictions."""
    names = [f"P{index + 3}" for index in range(len(feats))]
    ids = torch.cat(
        [
            torch.full((feature.shape[2] * feature.shape[3],), index, device=device, dtype=torch.long)
            for index, feature in enumerate(feats)
        ]
    )
    return ids.unsqueeze(0).expand(batch_size, -1), names


def summarize_distributions(
    store: dict[tuple[str, str], list[np.ndarray]],
    saturation_threshold: float,
    residual_variables: set[str],
    boundary_thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Convert raw distribution chunks into quantile rows."""
    boundary_thresholds = boundary_thresholds or {}
    rows = []
    for (scope, variable), chunks in sorted(store.items()):
        values = np.concatenate(chunks)
        boundary = saturation_threshold if variable in residual_variables else boundary_thresholds.get(variable)
        rows.append(
            {
                "scope": scope,
                "variable": variable,
                "n": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "p01": float(np.percentile(values, 1)),
                "p05": float(np.percentile(values, 5)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "max": float(values.max()),
                "saturation_rate": (
                    float((np.abs(values) >= boundary).mean()) if boundary is not None else np.nan
                ),
                "lower_boundary_rate": float((values <= -boundary).mean()) if boundary is not None else np.nan,
                "upper_boundary_rate": float((values >= boundary).mean()) if boundary is not None else np.nan,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "scope",
            "variable",
            "n",
            "mean",
            "std",
            "min",
            "p01",
            "p05",
            "p50",
            "p95",
            "p99",
            "max",
            "saturation_rate",
            "lower_boundary_rate",
            "upper_boundary_rate",
        ],
    )


def summarize_alignment(
    store: dict[tuple[str, str], dict[str, list[np.ndarray]]],
) -> pd.DataFrame:
    """Measure whether effective residuals track instance-level direct targets."""
    rows = []
    for (scope, channel), chunks in sorted(store.items()):
        prediction = np.concatenate(chunks["prediction"])
        target = np.concatenate(chunks["target"])
        pred_std = float(prediction.std())
        target_std = float(target.std())
        pearson = float(np.corrcoef(prediction, target)[0, 1]) if pred_std > 0 and target_std > 0 else np.nan
        rows.append(
            {
                "scope": scope,
                "channel": channel,
                "n": int(target.size),
                "prediction_mean": float(prediction.mean()),
                "prediction_std": pred_std,
                "target_mean": float(target.mean()),
                "target_std": target_std,
                "pearson_r": pearson,
                "direction_agreement": float((np.sign(prediction) == np.sign(target)).mean()),
                "mae": float(np.abs(prediction - target).mean()),
                "zero_baseline_mae": float(np.abs(target).mean()),
                "mean_baseline_mae": float(np.abs(target - target.mean()).mean()),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "scope",
            "channel",
            "n",
            "prediction_mean",
            "prediction_std",
            "target_mean",
            "target_std",
            "pearson_r",
            "direction_agreement",
            "mae",
            "zero_baseline_mae",
            "mean_baseline_mae",
        ],
    )


def summarize_gate_counts(
    counts: dict[tuple[str, str], dict[str, int]],
) -> pd.DataFrame:
    """Convert gate counters into a table."""
    rows = []
    for (scope, gate_name), values in sorted(counts.items()):
        total = values["total"]
        rows.append(
            {
                "scope": scope,
                "gate": gate_name,
                "active": values["active"],
                "total": total,
                "active_ratio": values["active"] / total if total else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=["scope", "gate", "active", "total", "active_ratio"])


def summarize_iou(
    store: dict[tuple[str, float, str], dict[str, list[np.ndarray]]],
) -> pd.DataFrame:
    """Convert paired IoU chunks into sweep statistics."""
    rows = []
    for (gate_name, alpha, subset), chunks in sorted(store.items()):
        coarse = np.concatenate(chunks["coarse"])
        refined = np.concatenate(chunks["refined"])
        delta = refined - coarse
        rows.append(
            {
                "gate": gate_name,
                "alpha": alpha,
                "subset": subset,
                "n": int(delta.size),
                "coarse_iou_mean": float(coarse.mean()),
                "refined_iou_mean": float(refined.mean()),
                "delta_iou_mean": float(delta.mean()),
                "delta_iou_p25": float(np.percentile(delta, 25)),
                "delta_iou_p50": float(np.percentile(delta, 50)),
                "delta_iou_p75": float(np.percentile(delta, 75)),
                "improved_ratio": float((delta > 1e-7).mean()),
                "worsened_ratio": float((delta < -1e-7).mean()),
                "unchanged_ratio": float((np.abs(delta) <= 1e-7).mean()),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "gate",
            "alpha",
            "subset",
            "n",
            "coarse_iou_mean",
            "refined_iou_mean",
            "delta_iou_mean",
            "delta_iou_p25",
            "delta_iou_p50",
            "delta_iou_p75",
            "improved_ratio",
            "worsened_ratio",
            "unchanged_ratio",
        ],
    )


def gradient_norms(
    core_model: torch.nn.Module,
    refine_head: torch.nn.Module,
    loss_vector: torch.Tensor,
) -> dict[str, float]:
    """Measure Refine/base gradient norms and the resulting global clipping scale."""
    refine_parameter_ids = {id(parameter) for parameter in refine_head.cv5.parameters()}
    base_squared = 0.0
    refine_squared = 0.0
    for parameter in core_model.parameters():
        if parameter.grad is None:
            continue
        squared = float(parameter.grad.detach().float().pow(2).sum().item())
        if id(parameter) in refine_parameter_ids:
            refine_squared += squared
        else:
            base_squared += squared
    base_norm = math.sqrt(base_squared)
    refine_norm = math.sqrt(refine_squared)
    global_norm = math.sqrt(base_squared + refine_squared)
    max_norm = 10.0
    return {
        "total_loss": float(loss_vector.detach().sum().item()),
        "base_grad_norm": base_norm,
        "refine_grad_norm": refine_norm,
        "global_grad_norm": global_norm,
        "base_only_clip_scale": min(1.0, max_norm / (base_norm + 1e-12)),
        "global_clip_scale": min(1.0, max_norm / (global_norm + 1e-12)),
    }


def markdown_table(dataframe: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
    """Render a compact Markdown table without requiring the optional tabulate package."""
    if dataframe.empty:
        return "无数据。"
    view = dataframe.loc[:, columns].head(max_rows).copy()
    for column in view.select_dtypes(include=[np.number]).columns:
        view[column] = view[column].map(lambda value: f"{value:.6f}" if pd.notna(value) else "")
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def append_training_log_summary(
    lines: list[str],
    training_diag: Path | None,
    training_results: Path | None,
) -> None:
    """Append optional training-time diagnostics to the Markdown report."""
    if training_diag is not None:
        frame = pd.read_csv(training_diag)
        lines.extend(["", "## 训练期 refine_diag.csv", ""])
        if frame.empty:
            lines.append("文件为空。")
        else:
            columns = [
                column
                for column in ("epoch", "refine_mask_ratio", "avg_abs_dshort", "refine_loss")
                if column in frame
            ]
            lines.append(markdown_table(frame.tail(10), columns, max_rows=10))
    if training_results is not None:
        frame = pd.read_csv(training_results)
        lines.extend(["", "## 训练 results.csv 的末尾趋势", ""])
        columns = [
            column
            for column in (
                "epoch",
                "train/aux_geo_loss",
                "val/aux_geo_loss",
                "metrics/mAP50-95(B)",
            )
            if column in frame
        ]
        lines.append(markdown_table(frame.tail(10), columns, max_rows=10))


def write_report(
    output_path: Path,
    args: argparse.Namespace,
    clamp_value: float,
    saturation_threshold: float,
    refine_version: str,
    experiment: str,
    target_limit: float | None,
    processed_batches: int,
    seen_images: int,
    gate_frame: pd.DataFrame,
    distribution_frame: pd.DataFrame,
    alignment_frame: pd.DataFrame,
    iou_frame: pd.DataFrame,
    confusion: dict[str, int],
    gradient_frame: pd.DataFrame,
) -> None:
    """Write a human-readable diagnostic report and decision rules."""
    lines = [
        "# Refine Head 机制诊断报告",
        "",
        "## 运行配置",
        "",
        f"- 权重：`{args.weights}`",
        f"- 数据：`{args.data}`",
        f"- split：`{args.split}`",
        f"- imgsz：`{args.imgsz}`",
        f"- 已处理 batch / image：`{processed_batches}` / `{seen_images}`",
        f"- 门控阈值：`AR>{args.ar_threshold}` 或 `short<{args.short_threshold}px`",
        f"- Refine版本 / 实验：`{refine_version}` / `{experiment}`",
        f"- residual limit / 饱和判定：`{clamp_value}` / `|delta|>={saturation_threshold}`",
        f"- direct target limit：`{target_limit if target_limit is not None else 'N/A'}`",
        f"- alpha：`{args.alphas}`",
        "",
        "本报告中的 IoU 是匹配正样本上的 ProbIoU 机制诊断，不等价于完整验证集 mAP；完整指标应由 "
        "`collect_refine_ab_curve.py` 生成。",
        "",
        "## 推理门控激活率",
        "",
    ]
    gate_scopes = gate_frame[
        gate_frame["scope"].isin(
            ["all_anchors", "P3_all", "P4_all", "P5_all", "positive_anchors", "confidence>=0.25"]
        )
    ]
    lines.append(markdown_table(gate_scopes, ["scope", "gate", "active", "total", "active_ratio"]))

    confusion_total = sum(confusion.values())
    lines.extend(["", "## GT 门控与预测门控的一致性", ""])
    if confusion_total:
        confusion_rows = pd.DataFrame(
            [
                {
                    "case": key,
                    "count": value,
                    "ratio": value / confusion_total,
                }
                for key, value in confusion.items()
            ]
        )
        lines.append(markdown_table(confusion_rows, ["case", "count", "ratio"]))
    else:
        lines.append("没有匹配正样本。")

    lines.extend(["", "## 当前门控的 alpha 扫描（匹配正样本）", ""])
    current = iou_frame[(iou_frame["gate"] == "current") & (iou_frame["subset"] == "all")]
    lines.append(
        markdown_table(
            current,
            [
                "gate",
                "alpha",
                "n",
                "coarse_iou_mean",
                "refined_iou_mean",
                "delta_iou_mean",
                "improved_ratio",
                "worsened_ratio",
            ],
        )
    )

    lines.extend(["", "## GT oracle 门控（区分门控错误与残差错误）", ""])
    oracle = iou_frame[(iou_frame["gate"] == "gt-oracle") & (iou_frame["subset"] == "all")]
    lines.append(
        markdown_table(
            oracle,
            [
                "gate",
                "alpha",
                "n",
                "coarse_iou_mean",
                "refined_iou_mean",
                "delta_iou_mean",
                "improved_ratio",
                "worsened_ratio",
            ],
        )
    )

    lines.extend(["", "## 裁剪目标的 scale-only oracle", ""])
    target_oracle = iou_frame[(iou_frame["gate"] == "target-oracle") & (iou_frame["subset"] == "all")]
    lines.append(
        markdown_table(
            target_oracle,
            [
                "gate",
                "alpha",
                "n",
                "coarse_iou_mean",
                "refined_iou_mean",
                "delta_iou_mean",
                "improved_ratio",
                "worsened_ratio",
            ],
        )
    )

    lines.extend(["", "## 残差分布重点项", ""])
    selected_distribution = distribution_frame[
        distribution_frame["scope"].isin(
            ["all_anchors", "current_gate", "positive_anchors", "gt_gated_positive", "pred_gated_positive"]
        )
    ]
    lines.append(
        markdown_table(
            selected_distribution,
            [
                "scope",
                "variable",
                "n",
                "mean",
                "std",
                "p05",
                "p50",
                "p95",
                "p99",
                "saturation_rate",
                "lower_boundary_rate",
                "upper_boundary_rate",
            ],
        )
    )

    target_distribution = distribution_frame[
        distribution_frame["variable"].astype(str).str.startswith("target_")
    ]
    if not target_distribution.empty:
        lines.extend(["", "## 短边/长边直接监督目标", ""])
        lines.append(
            markdown_table(
                target_distribution,
                [
                    "scope",
                    "variable",
                    "n",
                    "mean",
                    "std",
                    "p05",
                    "p50",
                    "p95",
                    "saturation_rate",
                    "lower_boundary_rate",
                    "upper_boundary_rate",
                ],
            )
        )

    if not alignment_frame.empty:
        lines.extend(["", "## 预测残差与直接监督目标的一致性", ""])
        lines.append(
            markdown_table(
                alignment_frame,
                [
                    "scope",
                    "channel",
                    "n",
                    "prediction_mean",
                    "prediction_std",
                    "target_mean",
                    "target_std",
                    "pearson_r",
                    "direction_agreement",
                    "mae",
                    "zero_baseline_mae",
                ],
            )
        )

    if not gradient_frame.empty:
        lines.extend(["", "## 梯度裁剪耦合", ""])
        lines.append(
            markdown_table(
                gradient_frame,
                [
                    "batch",
                    "total_loss",
                    "base_grad_norm",
                    "refine_grad_norm",
                    "global_grad_norm",
                    "base_only_clip_scale",
                    "global_clip_scale",
                ],
            )
        )

    lines.extend(
        [
            "",
            "## 判读规则",
            "",
            "1. `alpha=0` 的匹配正样本 ΔIoU 应为 0；完整 mAP 恒等性由 identity profile 检查。",
            "2. 小 alpha 恢复或改善、alpha=1 明显下降：残差方向可能正确，但幅度/校准失控。",
            "3. 所有 `alpha>0` 都下降：优先检查 residual target、宽高轴对应和损失定义。",
            "4. GT oracle 明显优于 current gate：训练/推理门控错配是主因。",
            "5. GT oracle 也明显下降：即使门控正确，Refine 学到的残差方向仍然错误。",
            "6. `pred_only` 比例高：推理在大量训练期未监督位置启用了 Refine，应加入 identity 约束或统一门控。",
            (
                "7. `saturation_rate` 应低于 5%；V2 按 `|delta|>=0.95*delta_max` 判定平滑边界饱和。"
                if refine_version != "legacy"
                else "7. legacy按 `|raw_delta|>=clamp` 判定硬截断饱和。"
            ),
            "8. target 边界率高：直接监督目标被大量裁剪，需先检查目标分布而不是继续缩小输出范围。",
            "9. 预测标准差接近 0、相关系数低或 MAE 不优于零基线：分支退化为常数修正。",
            "10. `global_clip_scale` 明显小于 `base_only_clip_scale`：cv5 通过全模型梯度裁剪间接影响 coarse 分支。",
        ]
    )
    append_training_log_summary(lines, args.training_diag, args.training_results)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Collect Refine mechanism diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=50, help="0 表示遍历完整 split")
    parser.add_argument("--ar-threshold", type=float, default=30.0)
    parser.add_argument("--short-threshold", type=float, default=16.0)
    parser.add_argument("--clamp", type=float, help="覆盖 legacy clamp 或 V2 delta_max；默认读取检测头")
    parser.add_argument(
        "--alphas",
        type=parse_alphas,
        default=parse_alphas("0,0.05,0.1,0.2,0.5,1"),
    )
    parser.add_argument(
        "--gate-modes",
        type=parse_gate_modes,
        default=parse_gate_modes("current,ar-only,short-only,and"),
    )
    parser.add_argument(
        "--conf-thresholds",
        type=parse_thresholds,
        default=parse_thresholds("0.001,0.01,0.1,0.25"),
    )
    parser.add_argument(
        "--gradient-batches",
        type=int,
        default=0,
        help="额外反向传播的 batch 数；用于检查 cv5 是否通过全局 clip 间接影响 coarse",
    )
    parser.add_argument("--training-diag", type=Path, help="可选：训练目录中的 refine_diag.csv")
    parser.add_argument("--training-results", type=Path, help="可选：训练目录中的 results.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.imgsz != 640:
        parser.error("创新点一固定使用 imgsz=640")
    if args.max_batches < 0 or args.gradient_batches < 0:
        parser.error("--max-batches 和 --gradient-batches 不能为负数")
    if args.clamp is not None and args.clamp <= 0:
        parser.error("--clamp 必须大于 0")
    if not args.weights.exists():
        raise FileNotFoundError(args.weights)
    for optional_path in (args.training_diag, args.training_results):
        if optional_path is not None and not optional_path.exists():
            raise FileNotFoundError(optional_path)

    global DEFAULT_CFG, DEFAULT_CFG_KEYS, YOLO
    global build_dataloader, build_yolo_dataset, check_det_dataset, get_cfg, make_anchors
    global np, pd, probiou, select_device, torch
    import numpy as np
    import pandas as pd
    import torch

    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_KEYS
    from ultralytics.utils.metrics import probiou
    from ultralytics.utils.tal import make_anchors
    from ultralytics.utils.torch_utils import select_device

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    yolo = YOLO(str(args.weights))
    core_model = yolo.model
    core_model.to(device).float().eval()
    for parameter in core_model.parameters():
        parameter.requires_grad_(True)

    refine_head = find_refine_head(core_model)
    is_v2 = getattr(refine_head, "refine_version", 1) == 2
    checkpoint_runtime_args = read_refine_runtime_args(refine_head) if is_v2 else {}
    cfg = configure_model_args(core_model, args, checkpoint_runtime_args)
    criterion = core_model.init_criterion()
    if is_v2:
        assert_refine_runtime_args(refine_head, checkpoint_runtime_args)

    default_limit = getattr(refine_head, "refine_delta_max", None) if is_v2 else None
    if default_limit is None:
        default_limit = getattr(refine_head, "refine_clamp", 1.0)
    clamp_value = float(args.clamp if args.clamp is not None else default_limit)
    if is_v2 and args.clamp is not None:
        refine_head.refine_delta_max = clamp_value
    experiment = getattr(refine_head, "refine_experiment", "legacy")
    target_limit = float(getattr(refine_head, "refine_target_limit", 0.0)) if is_v2 else None
    refine_version_label = (
        "v2.3"
        if experiment == "stable_aligned_gate"
        else "v2.2"
        if experiment == "stable_raw_short_long"
        else "v2.1"
        if experiment == "conservative_short_long"
        else "v2"
        if is_v2
        else "legacy"
    )
    channel_names = ("dw", "dh") if not is_v2 or experiment == "bounded_wh" else ("dshort", "dlong")
    raw_channel_names = tuple(f"raw_{name}" for name in channel_names)
    saturation_threshold = 0.95 * clamp_value if is_v2 else clamp_value
    residual_variables = set(channel_names) | {"dshort"}
    boundary_thresholds = (
        {
            "target_dshort": target_limit * (1.0 - 1e-6),
            "target_dlong": target_limit * (1.0 - 1e-6),
        }
        if target_limit is not None and target_limit > 0
        else {}
    )
    data_dict = check_det_dataset(args.data)
    stride = max(int(core_model.stride.max().item()), 32)
    loader = build_split_loader(cfg, data_dict, args.split, args.batch, args.workers, stride)

    distributions: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    alignments: dict[tuple[str, str], dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"prediction": [], "target": []}
    )
    gate_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"active": 0, "total": 0})
    iou_store: dict[tuple[str, float, str], dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"coarse": [], "refined": []}
    )
    confusion = {"both_on": 0, "pred_only": 0, "gt_only": 0, "both_off": 0}
    gradient_rows = []
    processed_batches = 0
    seen_images = 0

    print("=" * 80)
    print("Refine Head 机制诊断")
    print(f"weights={args.weights}")
    print(f"data={args.data}, split={args.split}, imgsz={args.imgsz}")
    print(
        f"AR>{args.ar_threshold}, short<{args.short_threshold}px, residual_limit={clamp_value}, "
        f"target_limit={target_limit}, version={refine_version_label}, experiment={experiment}"
    )
    print("=" * 80)

    for batch_index, raw_batch in enumerate(loader):
        if args.max_batches and batch_index >= args.max_batches:
            break
        batch = prepare_batch(raw_batch, device)
        batch_size = int(batch["img"].shape[0])
        processed_batches += 1
        seen_images += batch_size

        with torch.no_grad():
            raw = extract_raw_predictions(core_model(batch["img"]))
            assignment = build_assignments(criterion, raw, batch)

        coarse_px = assignment["coarse_px"]
        raw_residual = assignment["pred_refine"]
        residual = refine_head.bound_refine(raw_residual) if is_v2 else raw_residual
        delta_wh = residual_to_wh_delta(
            coarse_px,
            effective_residual(raw_residual, refine_head, clamp_value),
            refine_head,
        )
        pred_scores = assignment["pred_scores"].sigmoid().amax(dim=-1)
        fg_mask = assignment["fg_mask"]
        predicted_gates = build_predicted_gates(
            coarse_px,
            args.ar_threshold,
            args.short_threshold,
        )
        level_ids, level_names = fpn_level_ids(raw["feats"], batch_size, device)

        extend_values(distributions, "all_anchors", channel_names[0], residual[..., 0])
        extend_values(distributions, "all_anchors", channel_names[1], residual[..., 1])
        if is_v2:
            extend_values(distributions, "all_anchors", raw_channel_names[0], raw_residual[..., 0])
            extend_values(distributions, "all_anchors", raw_channel_names[1], raw_residual[..., 1])
        for level_index, level_name in enumerate(level_names):
            level_mask = level_ids == level_index
            extend_values(distributions, f"{level_name}_all", channel_names[0], residual[..., 0][level_mask])
            extend_values(distributions, f"{level_name}_all", channel_names[1], residual[..., 1][level_mask])
            if is_v2:
                extend_values(
                    distributions,
                    f"{level_name}_all",
                    raw_channel_names[0],
                    raw_residual[..., 0][level_mask],
                )
                extend_values(
                    distributions,
                    f"{level_name}_all",
                    raw_channel_names[1],
                    raw_residual[..., 1][level_mask],
                )

        for gate_name, gate in predicted_gates.items():
            add_gate_count(gate_counts, "all_anchors", gate_name, gate)
            for level_index, level_name in enumerate(level_names):
                add_gate_count(gate_counts, f"{level_name}_all", gate_name, gate, level_ids == level_index)

        current_gate = predicted_gates["current"]
        extend_values(distributions, "current_gate", channel_names[0], residual[..., 0][current_gate])
        extend_values(distributions, "current_gate", channel_names[1], residual[..., 1][current_gate])
        extend_values(
            distributions,
            "current_gate",
            "scale_w_alpha1",
            delta_wh[..., 0][current_gate].exp(),
        )
        extend_values(
            distributions,
            "current_gate",
            "scale_h_alpha1",
            delta_wh[..., 1][current_gate].exp(),
        )
        for threshold in args.conf_thresholds:
            confidence_mask = pred_scores >= threshold
            scope = f"confidence>={threshold:g}"
            add_gate_count(gate_counts, scope, "current", current_gate, confidence_mask)
            selected = confidence_mask & current_gate
            extend_values(distributions, scope, channel_names[0], residual[..., 0][selected])
            extend_values(distributions, scope, channel_names[1], residual[..., 1][selected])

        if fg_mask.any():
            coarse_fg = coarse_px[fg_mask]
            raw_residual_fg = raw_residual[fg_mask]
            residual_fg = residual[fg_mask]
            target_fg = assignment["target_bboxes"][fg_mask]
            positive_level_ids = level_ids[fg_mask]
            gt_gate, gt_short, gt_ar = build_gt_gate(
                target_fg,
                args.ar_threshold,
                args.short_threshold,
            )
            predicted_positive_gates = {name: gate[fg_mask] for name, gate in predicted_gates.items()}
            predicted_current = predicted_positive_gates["current"]

            direct_target_delta = None
            if is_v2 and experiment != "bounded_wh":
                coarse_short = coarse_fg[:, 2:4].amin(dim=-1)
                coarse_long = coarse_fg[:, 2:4].amax(dim=-1)
                target_short = target_fg[:, 2:4].amin(dim=-1)
                target_long = target_fg[:, 2:4].amax(dim=-1)
                eps = 1e-6
                direct_target_raw = torch.stack(
                    (
                        torch.log((target_short + eps) / (coarse_short + eps)),
                        torch.log((target_long + eps) / (coarse_long + eps)),
                    ),
                    dim=-1,
                )
                direct_target_delta = direct_target_raw.clamp(-target_limit, target_limit)
                target_scopes = {
                    "positive_anchors": torch.ones_like(gt_gate),
                    "gt_gated_positive": gt_gate,
                    "pred_gated_positive": predicted_current,
                }
                for scope, scope_mask in target_scopes.items():
                    for channel_index, channel_name in enumerate(("dshort", "dlong")):
                        extend_values(
                            distributions,
                            scope,
                            f"target_raw_{channel_name}",
                            direct_target_raw[:, channel_index][scope_mask],
                        )
                        extend_values(
                            distributions,
                            scope,
                            f"target_{channel_name}",
                            direct_target_delta[:, channel_index][scope_mask],
                        )
                        extend_alignment(
                            alignments,
                            scope,
                            channel_name,
                            residual_fg[:, channel_index][scope_mask],
                            direct_target_delta[:, channel_index][scope_mask],
                        )
                for level_index, level_name in enumerate(level_names):
                    level_gt_gated = (positive_level_ids == level_index) & gt_gate
                    for channel_index, channel_name in enumerate(("dshort", "dlong")):
                        extend_alignment(
                            alignments,
                            f"{level_name}_gt_gated",
                            channel_name,
                            residual_fg[:, channel_index][level_gt_gated],
                            direct_target_delta[:, channel_index][level_gt_gated],
                        )

            add_gate_count(gate_counts, "positive_anchors", "pred_current", predicted_current)
            add_gate_count(gate_counts, "positive_anchors", "gt_oracle", gt_gate)
            for level_index, level_name in enumerate(level_names):
                level_positive = positive_level_ids == level_index
                add_gate_count(gate_counts, f"{level_name}_positive", "pred_current", predicted_current, level_positive)
                add_gate_count(gate_counts, f"{level_name}_positive", "gt_oracle", gt_gate, level_positive)

            confusion["both_on"] += int((predicted_current & gt_gate).sum().item())
            confusion["pred_only"] += int((predicted_current & ~gt_gate).sum().item())
            confusion["gt_only"] += int((~predicted_current & gt_gate).sum().item())
            confusion["both_off"] += int((~predicted_current & ~gt_gate).sum().item())

            extend_values(distributions, "positive_anchors", channel_names[0], residual_fg[:, 0])
            extend_values(distributions, "positive_anchors", channel_names[1], residual_fg[:, 1])
            extend_values(distributions, "gt_gated_positive", channel_names[0], residual_fg[:, 0][gt_gate])
            extend_values(distributions, "gt_gated_positive", channel_names[1], residual_fg[:, 1][gt_gate])
            extend_values(
                distributions,
                "pred_gated_positive",
                channel_names[0],
                residual_fg[:, 0][predicted_current],
            )
            extend_values(
                distributions,
                "pred_gated_positive",
                channel_names[1],
                residual_fg[:, 1][predicted_current],
            )
            if is_v2 and experiment != "bounded_wh":
                dshort = residual_fg[:, 0]
            elif is_v2:
                short_is_width = coarse_fg[:, 2] <= coarse_fg[:, 3]
                dshort = torch.where(short_is_width, residual_fg[:, 0], residual_fg[:, 1])
            else:
                short_is_width = coarse_fg[:, 2] <= coarse_fg[:, 3]
                dshort = torch.where(short_is_width, raw_residual_fg[:, 0], raw_residual_fg[:, 1])
            if channel_names[0] != "dshort":
                extend_values(distributions, "positive_anchors", "dshort", dshort)
                extend_values(distributions, "gt_gated_positive", "dshort", dshort[gt_gate])
                extend_values(distributions, "pred_gated_positive", "dshort", dshort[predicted_current])
            extend_values(
                distributions,
                "gt_gated_positive",
                "scale_short_alpha1",
                (
                    dshort[gt_gate].exp()
                    if is_v2
                    else dshort[gt_gate].clamp(-clamp_value, clamp_value).exp()
                ),
            )
            extend_values(
                distributions,
                "pred_gated_positive",
                "scale_short_alpha1",
                (
                    dshort[predicted_current].exp()
                    if is_v2
                    else dshort[predicted_current].clamp(-clamp_value, clamp_value).exp()
                ),
            )

            coarse_iou = probiou(coarse_fg, target_fg).reshape(-1)
            if direct_target_delta is not None:
                target_oracle_boxes = apply_short_long_target(
                    coarse_fg,
                    direct_target_delta,
                    gt_gate,
                    refine_head,
                )
                target_oracle_iou = probiou(target_oracle_boxes, target_fg).reshape(-1)
                add_iou_values(
                    iou_store,
                    "target-oracle",
                    1.0,
                    "all",
                    coarse_iou,
                    target_oracle_iou,
                )
            sweep_gates = {
                gate_name: predicted_positive_gates[gate_name]
                for gate_name in args.gate_modes
            }
            sweep_gates["gt-oracle"] = gt_gate
            for gate_name, gate in sweep_gates.items():
                for alpha in args.alphas:
                    refined = apply_refine(
                        coarse_fg,
                        raw_residual_fg,
                        gate,
                        alpha,
                        clamp_value,
                        refine_head,
                    )
                    refined_iou = probiou(refined, target_fg).reshape(-1)
                    add_iou_values(iou_store, gate_name, alpha, "all", coarse_iou, refined_iou)

                    if alpha == 1.0 and gate_name in {"current", "gt-oracle"}:
                        subgroups = {
                            "gt_gated": gt_gate,
                            "gt_not_gated": ~gt_gate,
                            f"short_lt_{args.short_threshold:g}": gt_short < args.short_threshold,
                            f"short_{args.short_threshold:g}_to_32": (
                                (gt_short >= args.short_threshold) & (gt_short < 32.0)
                            ),
                            "short_ge_32": gt_short >= 32.0,
                            f"ar_gt_{args.ar_threshold:g}": gt_ar > args.ar_threshold,
                            f"ar_le_{args.ar_threshold:g}": gt_ar <= args.ar_threshold,
                        }
                        for level_index, level_name in enumerate(level_names):
                            subgroups[level_name] = positive_level_ids == level_index
                        for subgroup, subgroup_mask in subgroups.items():
                            add_iou_values(
                                iou_store,
                                gate_name,
                                alpha,
                                subgroup,
                                coarse_iou,
                                refined_iou,
                                subgroup_mask,
                            )

        if batch_index < args.gradient_batches:
            core_model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                outputs_for_gradient = core_model(batch["img"])
                loss_vector, loss_items = criterion(outputs_for_gradient, batch)
                loss_vector.sum().backward()
            row = {
                "batch": batch_index,
                **gradient_norms(core_model, refine_head, loss_vector),
            }
            for item_index, item_name in enumerate(("box", "cls", "dfl", "angle", "aux_geo")):
                if item_index < loss_items.numel():
                    row[f"{item_name}_loss"] = float(loss_items[item_index].item())
            gradient_rows.append(row)
            core_model.zero_grad(set_to_none=True)

        if processed_batches % 10 == 0:
            print(f"已处理 {processed_batches} batches / {seen_images} images")

    distribution_frame = summarize_distributions(
        distributions,
        saturation_threshold,
        residual_variables,
        boundary_thresholds,
    )
    alignment_frame = summarize_alignment(alignments)
    gate_frame = summarize_gate_counts(gate_counts)
    iou_frame = summarize_iou(iou_store)
    gradient_frame = pd.DataFrame(gradient_rows)

    distribution_frame.to_csv(args.output_dir / "refine_distribution.csv", index=False, encoding="utf-8-sig")
    if not alignment_frame.empty:
        alignment_frame.to_csv(args.output_dir / "refine_target_alignment.csv", index=False, encoding="utf-8-sig")
    gate_frame.to_csv(args.output_dir / "refine_gate_stats.csv", index=False, encoding="utf-8-sig")
    iou_frame.to_csv(args.output_dir / "refine_iou_sweep.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "case": key,
                "count": value,
                "ratio": value / max(sum(confusion.values()), 1),
            }
            for key, value in confusion.items()
        ]
    ).to_csv(args.output_dir / "refine_gate_confusion.csv", index=False, encoding="utf-8-sig")
    if not gradient_frame.empty:
        gradient_frame.to_csv(args.output_dir / "refine_gradient_stats.csv", index=False, encoding="utf-8-sig")

    report_path = args.output_dir / "refine_diagnostics.md"
    write_report(
        report_path,
        args,
        clamp_value,
        saturation_threshold,
        refine_version_label,
        experiment,
        target_limit,
        processed_batches,
        seen_images,
        gate_frame,
        distribution_frame,
        alignment_frame,
        iou_frame,
        confusion,
        gradient_frame,
    )
    print("=" * 80)
    print(f"诊断完成：{args.output_dir}")
    print(f"人工判读入口：{report_path}")


if __name__ == "__main__":
    main()
