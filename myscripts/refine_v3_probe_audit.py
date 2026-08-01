"""Audit Refine V3 residual predictability with train-only model selection.

This is a second-stage diagnostic, not a new Refine head and not a test-set
evaluation.  It corrects two limitations of the first precheck:

* the original residual probe did not report fit/holdout convergence;
* it tested only short/long scale residuals under the legacy current gate.

The audit fits a closed-form ridge probe and a target-normalized MLP on a fixed
train-fit split, selects hyperparameters only on a train-holdout split, and then
evaluates val once.  Scale, local center and long-axis angle residuals are tested
independently, both on all matched positives and (for scale) the legacy gate.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any, Sequence

import myscripts.refine_v3_precheck as v3


TARGET_CHANNELS = {
    "scale": ("dshort", "dlong"),
    "center": ("dcenter_long", "dcenter_short"),
    "angle": ("dtheta",),
}
TASKS = (
    ("scale", "all_positive"),
    ("scale", "current_gate"),
    ("center", "all_positive"),
    ("angle", "all_positive"),
)


def parse_float_csv(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("必须提供一个或多个非负浮点数")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = v3.build_parser()
    parser.description = __doc__
    parser.set_defaults(probe_epochs=60, probe_hidden=96)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-epochs", type=int, default=10)
    parser.add_argument("--ridge-lambdas", type=parse_float_csv, default=parse_float_csv("0,1e-4,1e-3,1e-2,1e-1,1,10"))
    parser.add_argument("--scale-target-limit", type=float, default=0.1)
    parser.add_argument("--center-target-limit", type=float, default=2.0, help="单位为对应 FPN stride")
    parser.add_argument("--direction-deadzone", type=float, default=1e-3)
    parser.add_argument("--probe-min-samples", type=int, default=64)
    parser.add_argument("--probe-min-delta-iou", type=float, default=0.0)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    v3.validate_args(parser, args)
    if not 0.05 <= args.holdout_fraction <= 0.5:
        parser.error("--holdout-fraction 必须位于 [0.05, 0.5]")
    if args.early_stop_patience <= 0 or args.early_stop_min_epochs <= 0:
        parser.error("early-stop 参数必须大于 0")
    if args.early_stop_min_epochs > args.probe_epochs:
        parser.error("--early-stop-min-epochs 不能大于 --probe-epochs")
    if args.scale_target_limit <= 0 or args.center_target_limit <= 0:
        parser.error("target limit 必须大于 0")
    if args.direction_deadzone < 0 or args.probe_min_samples < 16:
        parser.error("deadzone 不能为负且 probe 最少样本数不能小于 16")


def stable_seed(base: int, *parts: str) -> int:
    value = int(base)
    for part in parts:
        for character in part:
            value = (value * 131 + ord(character)) % 2_147_483_647
    return value


def scope_mask(data: dict[str, Any], scope: str):
    if scope == "all_positive":
        return torch.ones(data["current_gate"].shape[0], dtype=torch.bool)
    if scope == "current_gate":
        return data["current_gate"].bool()
    raise ValueError(scope)


def target_tensor(data: dict[str, Any], family: str, args: argparse.Namespace):
    if family == "scale":
        return data["scale_target_exact"].float().clamp(-args.scale_target_limit, args.scale_target_limit)
    if family == "center":
        return data["center_target"].float().clamp(-args.center_target_limit, args.center_target_limit)
    if family == "angle":
        return data["angle_target"].float().clamp(-1.0, 1.0)
    raise ValueError(family)


def feature_tensor(data: dict[str, Any], feature_set: str):
    return v3.select_features(data, feature_set).float()


def fixed_fit_holdout_indices(length: int, fraction: float, seed: int):
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(length, generator=generator)
    holdout_count = max(1, min(length - 1, int(round(length * fraction))))
    return order[holdout_count:], order[:holdout_count]


def standardize_fit_features(x_fit):
    mean = x_fit.mean(dim=0)
    std = x_fit.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


def predict_in_batches(model, x, x_mean, x_std, *, device, batch_size: int):
    outputs = []
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            xb = ((x[start : start + batch_size] - x_mean) / x_std).to(device)
            outputs.append(model(xb).detach().cpu())
    return torch.cat(outputs, dim=0)


def fit_ridge_probe(x_fit, y_fit, x_holdout, y_holdout, *, lambdas: Sequence[float], device):
    """Select ridge strength on train-holdout; val is never visible here."""
    x_mean, x_std = standardize_fit_features(x_fit)
    y_mean = y_fit.mean(dim=0, keepdim=True)
    x_device = ((x_fit - x_mean) / x_std).to(device)
    y_device = (y_fit - y_mean).to(device)
    scale = 1.0 / max(int(x_fit.shape[0]), 1)
    gram = x_device.T @ x_device * scale
    cross = x_device.T @ y_device * scale
    identity = torch.eye(gram.shape[0], device=device, dtype=gram.dtype)
    best = None
    for ridge_lambda in lambdas:
        try:
            weights = torch.linalg.solve(gram + float(ridge_lambda) * identity, cross)
        except RuntimeError:
            continue
        holdout_prediction = (((x_holdout - x_mean) / x_std).to(device) @ weights).cpu() + y_mean
        holdout_mae = float((holdout_prediction - y_holdout).abs().mean().item())
        if best is None or holdout_mae < best[0]:
            best = (holdout_mae, float(ridge_lambda), weights.detach().cpu())
    if best is None:
        raise RuntimeError("所有 ridge 线性系统求解均失败")
    return {
        "kind": "ridge",
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "weights": best[2],
        "selected": best[1],
        "selected_name": "ridge_lambda",
    }


def build_mlp(input_dim: int, output_dim: int, hidden: int):
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden, output_dim),
    )
    torch.nn.init.zeros_(model[-1].weight)
    torch.nn.init.zeros_(model[-1].bias)
    return model


def fit_mlp_probe(x_fit, y_fit, x_holdout, y_holdout, *, args: argparse.Namespace, device, seed: int):
    """Use target normalization and train-holdout early stopping to audit convergence."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    x_mean, x_std = standardize_fit_features(x_fit)
    y_mean = y_fit.mean(dim=0, keepdim=True)
    y_std = y_fit.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
    model = build_mlp(x_fit.shape[1], y_fit.shape[1], args.probe_hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.probe_lr, weight_decay=1e-4)
    best_mae = math.inf
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    generator = torch.Generator().manual_seed(seed + 1)

    for epoch in range(1, args.probe_epochs + 1):
        model.train()
        order = torch.randperm(x_fit.shape[0], generator=generator)
        for start in range(0, x_fit.shape[0], args.probe_batch_size):
            indices = order[start : start + args.probe_batch_size]
            xb = ((x_fit[indices] - x_mean) / x_std).to(device)
            yb = ((y_fit[indices] - y_mean) / y_std).to(device)
            prediction = model(xb)
            loss = torch.nn.functional.smooth_l1_loss(prediction, yb, beta=0.2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        normalized = predict_in_batches(
            model, x_holdout, x_mean, x_std, device=device, batch_size=args.probe_batch_size
        )
        holdout_prediction = normalized * y_std + y_mean
        holdout_mae = float((holdout_prediction - y_holdout).abs().mean().item())
        if holdout_mae < best_mae - 1e-7:
            best_mae = holdout_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch >= args.early_stop_min_epochs and stale_epochs >= args.early_stop_patience:
            break

    if best_state is None:
        raise RuntimeError("MLP 未产生有效 checkpoint")
    model.load_state_dict(best_state)
    return {
        "kind": "mlp",
        "model": model.eval(),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "selected": best_epoch,
        "selected_name": "best_epoch",
    }


def probe_predict(fitted: dict[str, Any], x, *, device, batch_size: int):
    if fitted["kind"] == "ridge":
        return ((x - fitted["x_mean"]) / fitted["x_std"]) @ fitted["weights"] + fitted["y_mean"]
    normalized = predict_in_batches(
        fitted["model"], x, fitted["x_mean"], fitted["x_std"], device=device, batch_size=batch_size
    )
    return normalized * fitted["y_std"] + fitted["y_mean"]


def clip_prediction(prediction, family: str, args: argparse.Namespace):
    if family == "scale":
        return prediction.clamp(-args.scale_target_limit, args.scale_target_limit)
    if family == "center":
        return prediction.clamp(-args.center_target_limit, args.center_target_limit)
    if family == "angle":
        return prediction.clamp(-1.0, 1.0)
    raise ValueError(family)


def apply_probe_geometry(coarse, prediction, stride, family: str):
    refined = coarse.clone()
    if family == "scale":
        short_is_width = coarse[:, 2] <= coarse[:, 3]
        delta_width = torch.where(short_is_width, prediction[:, 0], prediction[:, 1])
        delta_height = torch.where(short_is_width, prediction[:, 1], prediction[:, 0])
        refined[:, 2] *= torch.exp(delta_width)
        refined[:, 3] *= torch.exp(delta_height)
    elif family == "center":
        long_angle = coarse[:, 4] + torch.where(
            coarse[:, 3] > coarse[:, 2],
            torch.full_like(coarse[:, 4], math.pi / 2.0),
            torch.zeros_like(coarse[:, 4]),
        )
        cos_long = torch.cos(long_angle)
        sin_long = torch.sin(long_angle)
        long_offset = prediction[:, 0] * stride.reshape(-1)
        short_offset = prediction[:, 1] * stride.reshape(-1)
        refined[:, 0] += long_offset * cos_long - short_offset * sin_long
        refined[:, 1] += long_offset * sin_long + short_offset * cos_long
    elif family == "angle":
        refined[:, 4] += prediction[:, 0] * (math.pi / 2.0)
    else:
        raise ValueError(family)
    return refined


def pearson(first, second) -> float:
    return v3.pearson_correlation(first, second)


def metric_rows(
    *,
    family: str,
    scope: str,
    features: str,
    model_kind: str,
    level: str,
    split: str,
    target,
    prediction,
    train_mean,
    selected_name: str,
    selected_value: float,
    args: argparse.Namespace,
    geometry: dict[str, Any] | None = None,
):
    rows = []
    channel_specs = [(name, index) for index, name in enumerate(TARGET_CHANNELS[family])]
    if len(channel_specs) > 1:
        channel_specs.append(("all", None))
    else:
        channel_specs = [("all", None)]
    for channel, index in channel_specs:
        actual = target if index is None else target[:, index : index + 1]
        predicted = prediction if index is None else prediction[:, index : index + 1]
        baseline = train_mean.expand_as(target) if index is None else train_mean[:, index : index + 1].expand_as(actual)
        mae = float((predicted - actual).abs().mean().item())
        zero_mae = float(actual.abs().mean().item())
        mean_mae = float((baseline - actual).abs().mean().item())
        best_baseline = min(zero_mae, mean_mae)
        relative_improvement = (best_baseline - mae) / max(best_baseline, 1e-12)
        eligible = actual.abs() > args.direction_deadzone
        direction = float(((predicted[eligible] * actual[eligible]) > 0).float().mean().item()) if eligible.any() else math.nan
        is_aggregate_channel = channel == "all"
        rows.append({
            "family": family,
            "scope": scope,
            "features": features,
            "model": model_kind,
            "level": level,
            "split": split,
            "channel": channel,
            "n": int(actual.numel()),
            "mae": mae,
            "zero_baseline_mae": zero_mae,
            "mean_baseline_mae": mean_mae,
            "relative_mae_improvement": relative_improvement,
            "direction_agreement": direction,
            "pearson_r": pearson(predicted, actual),
            "delta_iou_mean": geometry["delta_iou_mean"] if geometry and is_aggregate_channel else math.nan,
            "improved_ratio": geometry["improved_ratio"] if geometry and is_aggregate_channel else math.nan,
            "worsened_ratio": geometry["worsened_ratio"] if geometry and is_aggregate_channel else math.nan,
            "selected_hyperparameter": selected_name,
            "selected_value": selected_value,
            "metric_pass": bool(
                relative_improvement >= args.residual_min_mae_improvement
                and direction >= args.residual_min_direction
                and (not geometry or geometry["delta_iou_mean"] >= args.probe_min_delta_iou)
            ),
        })
    return rows


def geometry_metrics(coarse, target_box, stride, prediction, family: str):
    refined = apply_probe_geometry(coarse, prediction, stride, family)
    coarse_iou = v3.paired_iou(coarse, target_box)
    refined_iou = v3.paired_iou(refined, target_box)
    delta = refined_iou - coarse_iou
    return {
        "delta_iou_mean": float(delta.mean().item()),
        "improved_ratio": float((delta > 1e-7).float().mean().item()),
        "worsened_ratio": float((delta < -1e-7).float().mean().item()),
    }


def subset(data: dict[str, Any], indices):
    return {name: value[indices] for name, value in data.items()}


def run_probe_audit(train_data, eval_data, *, args: argparse.Namespace, device):
    rows = []
    for family, scope in TASKS:
        for features in ("state", "fpn_state"):
            for model_kind in ("ridge", "mlp"):
                aggregates = {name: {"target": [], "prediction": [], "mean": [], "coarse": [], "target_box": [], "stride": []} for name in ("fit", "holdout", "val")}
                for level in sorted(set(train_data).intersection(eval_data)):
                    train_level = train_data[level]
                    eval_level = eval_data[level]
                    train_indices = torch.where(scope_mask(train_level, scope))[0]
                    eval_indices = torch.where(scope_mask(eval_level, scope))[0]
                    if train_indices.numel() < args.probe_min_samples or eval_indices.numel() < 2:
                        continue
                    train_subset = subset(train_level, train_indices)
                    eval_subset = subset(eval_level, eval_indices)
                    fit_local, holdout_local = fixed_fit_holdout_indices(
                        train_indices.numel(), args.holdout_fraction,
                        stable_seed(args.seed, family, scope, level),
                    )
                    x_all = feature_tensor(train_subset, features)
                    y_all = target_tensor(train_subset, family, args)
                    x_fit, y_fit = x_all[fit_local], y_all[fit_local]
                    x_holdout, y_holdout = x_all[holdout_local], y_all[holdout_local]
                    x_val = feature_tensor(eval_subset, features)
                    y_val = target_tensor(eval_subset, family, args)
                    if model_kind == "ridge":
                        fitted = fit_ridge_probe(
                            x_fit, y_fit, x_holdout, y_holdout, lambdas=args.ridge_lambdas, device=device
                        )
                    else:
                        fitted = fit_mlp_probe(
                            x_fit, y_fit, x_holdout, y_holdout, args=args, device=device,
                            seed=stable_seed(args.seed, family, scope, features, level, model_kind),
                        )
                    train_mean = y_fit.mean(dim=0, keepdim=True)
                    split_payloads = {
                        "fit": (x_fit, y_fit, subset(train_subset, fit_local)),
                        "holdout": (x_holdout, y_holdout, subset(train_subset, holdout_local)),
                        "val": (x_val, y_val, eval_subset),
                    }
                    for split_name, (x_split, y_split, raw_split) in split_payloads.items():
                        prediction = clip_prediction(
                            probe_predict(fitted, x_split, device=device, batch_size=args.probe_batch_size), family, args
                        )
                        geometry = geometry_metrics(
                            raw_split["coarse_box"], raw_split["target_box"], raw_split["stride"], prediction, family
                        )
                        rows.extend(metric_rows(
                            family=family, scope=scope, features=features, model_kind=model_kind,
                            level=level, split=split_name, target=y_split, prediction=prediction,
                            train_mean=train_mean, selected_name=fitted["selected_name"],
                            selected_value=float(fitted["selected"]), args=args, geometry=geometry,
                        ))
                        aggregate = aggregates[split_name]
                        aggregate["target"].append(y_split)
                        aggregate["prediction"].append(prediction)
                        aggregate["mean"].append(train_mean.expand_as(y_split))
                        aggregate["coarse"].append(raw_split["coarse_box"])
                        aggregate["target_box"].append(raw_split["target_box"])
                        aggregate["stride"].append(raw_split["stride"])
                    if fitted["kind"] == "mlp":
                        del fitted["model"]

                for split_name, aggregate in aggregates.items():
                    if not aggregate["target"]:
                        continue
                    target = torch.cat(aggregate["target"])
                    prediction = torch.cat(aggregate["prediction"])
                    mean_baseline = torch.cat(aggregate["mean"])
                    geometry = geometry_metrics(
                        torch.cat(aggregate["coarse"]), torch.cat(aggregate["target_box"]),
                        torch.cat(aggregate["stride"]), prediction, family,
                    )
                    rows.extend(metric_rows(
                        family=family, scope=scope, features=features, model_kind=model_kind,
                        level="all", split=split_name, target=target, prediction=prediction,
                        train_mean=mean_baseline, selected_name="per_level", selected_value=math.nan,
                        args=args, geometry=geometry,
                    ))
                print(f"  finished family={family}, scope={scope}, features={features}, model={model_kind}")
    return pd.DataFrame(rows)


def summarize_candidates(metrics, oracle_frame, args: argparse.Namespace):
    candidate_columns = [
        "family", "scope", "features", "model", "oracle_gain",
        "holdout_relative_mae_improvement", "holdout_direction_agreement",
        "val_relative_mae_improvement", "val_direction_agreement", "val_delta_iou_mean",
        "val_improved_ratio", "val_worsened_ratio", "passing_levels", "pass",
    ]
    if metrics.empty:
        return pd.DataFrame(columns=candidate_columns)
    aggregate = metrics[(metrics["level"] == "all") & (metrics["channel"] == "all")]
    rows = []
    oracle_lookup = oracle_frame.set_index("variant")["delta_iou_mean"]
    oracle_gain = {
        "scale": float(oracle_lookup["scale_all"]),
        "center": float(oracle_lookup["center_all"]),
        "angle": float(oracle_lookup["angle_all"]),
    }
    keys = aggregate[["family", "scope", "features", "model"]].drop_duplicates()
    for key in keys.itertuples(index=False):
        selected_all = metrics[
            (metrics["family"] == key.family)
            & (metrics["scope"] == key.scope)
            & (metrics["features"] == key.features)
            & (metrics["model"] == key.model)
        ]
        selected_aggregate = selected_all[(selected_all["level"] == "all") & (selected_all["channel"] == "all")]
        holdout = selected_aggregate[selected_aggregate["split"] == "holdout"]
        val = selected_aggregate[selected_aggregate["split"] == "val"]
        if holdout.empty or val.empty:
            continue
        holdout_row = holdout.iloc[0]
        val_row = val.iloc[0]
        level_passes = []
        for level in sorted(set(selected_all["level"]).difference({"all"})):
            level_holdout = selected_all[
                (selected_all["level"] == level)
                & (selected_all["split"] == "holdout")
                & (selected_all["channel"] == "all")
            ]
            level_val = selected_all[
                (selected_all["level"] == level)
                & (selected_all["split"] == "val")
                & (selected_all["channel"] == "all")
            ]
            if not level_holdout.empty and not level_val.empty and bool(level_holdout.iloc[0]["metric_pass"]) and bool(level_val.iloc[0]["metric_pass"]):
                level_passes.append(level)
        passed = bool(
            holdout_row["metric_pass"]
            and val_row["metric_pass"]
            and oracle_gain[key.family] >= args.extra_dof_min_gain
            and level_passes
        )
        rows.append({
            "family": key.family,
            "scope": key.scope,
            "features": key.features,
            "model": key.model,
            "oracle_gain": oracle_gain[key.family],
            "holdout_relative_mae_improvement": holdout_row["relative_mae_improvement"],
            "holdout_direction_agreement": holdout_row["direction_agreement"],
            "val_relative_mae_improvement": val_row["relative_mae_improvement"],
            "val_direction_agreement": val_row["direction_agreement"],
            "val_delta_iou_mean": val_row["delta_iou_mean"],
            "val_improved_ratio": val_row["improved_ratio"],
            "val_worsened_ratio": val_row["worsened_ratio"],
            "passing_levels": ",".join(level_passes),
            "pass": passed,
        })
    return pd.DataFrame(rows, columns=candidate_columns).sort_values(
        ["pass", "val_delta_iou_mean", "val_relative_mae_improvement"], ascending=False
    )


def dataframe_to_markdown(frame, columns, max_rows: int = 30) -> str:
    return v3.dataframe_to_markdown(frame, columns, max_rows=max_rows)


def write_report(path: Path, *, args, audit_frame, oracle_frame, metrics, candidates, split_stats):
    passing = candidates[candidates["pass"]]
    best = candidates.head(12)
    lines = [
        "# Refine V3 残差 Probe 审计报告",
        "",
        "## 1. 目的与边界",
        "",
        "本报告用于区分‘残差不可预测’与‘原 Probe 未收敛’，并测试 scale、中心和长轴角度三类自由度。模型选择只使用 train-fit/train-holdout，val 只评估一次，test 未读取。",
        "",
        f"- 纯 CA：`{args.ca_weights}`",
        f"- Refine 特征 checkpoint：`{args.refine_weights}`",
        f"- train batches/images：`{split_stats[0]['batches']}` / `{split_stats[0]['images']}`",
        f"- val batches/images：`{split_stats[1]['batches']}` / `{split_stats[1]['images']}`",
        f"- CA 共享状态审计失败数：`{len(audit_frame[~audit_frame['status'].isin(('equal', 'allowed'))])}`",
        "",
        "## 2. 已知前置证据",
        "",
        dataframe_to_markdown(oracle_frame, ["variant", "n", "coarse_iou_mean", "refined_iou_mean", "delta_iou_mean", "improved_ratio", "worsened_ratio"]),
        "",
        "第一阶段结果中，oracle scale 标签可分类，但 V2.2 实际残差收益的聚合 AUC 未达 0.65，原 scale 残差回归也没有超过零/均值基线。因此本轮不实现 Head，只审计更强 Probe 与新增自由度。",
        "",
        "## 3. 聚合候选",
        "",
        dataframe_to_markdown(best, [
            "family", "scope", "features", "model", "oracle_gain",
            "holdout_relative_mae_improvement", "holdout_direction_agreement",
            "val_relative_mae_improvement", "val_direction_agreement", "val_delta_iou_mean",
            "val_improved_ratio", "val_worsened_ratio", "pass",
            "passing_levels",
        ]),
        "",
        "## 4. 冻结判定",
        "",
    ]
    if passing.empty:
        lines.extend([
            "没有自由度同时通过 train-holdout、val 和 oracle 三项门槛。停止 Refine 结构扩张，论文主线回退为 Coverage-Aware + reg_max=32；Refine 仅保留为负结果/局限性分析。",
        ])
    else:
        families = ", ".join(sorted(passing["family"].unique()))
        lines.extend([
            f"通过的自由度：`{families}`。下一步只为第一个通过的最小自由度及其 `passing_levels` 实现独立 V3 Head；其余自由度和层级不同时加入。",
            "V3 seed0 学到实际残差后，再用该残差定义质量收益标签并训练/验证质量门控；当前 oracle 标签不能替代部署门控证据。",
        ])
    lines.extend([
        "",
        "## 5. 输出",
        "",
        "- `checkpoint_audit.csv`：共享状态审计。",
        "- `oracle_dof.csv`：第一阶段 oracle 口径复算。",
        "- `probe_metrics.csv`：各层及聚合的 fit/holdout/val 指标。",
        "- `candidate_summary.csv`：用于决策的聚合候选。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    global np, pd, torch
    rd, YOLO, check_det_dataset, make_anchors, select_device = v3.bind_runtime_dependencies()
    np, pd, torch = v3.np, v3.pd, v3.torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)

    print("=" * 80)
    print("Refine V3 probe audit: train-fit -> train-holdout selection -> val once")
    print(f"CA={args.ca_weights}")
    print(f"Refine={args.refine_weights}")
    print("=" * 80)

    ca_yolo = YOLO(str(args.ca_weights), task="obb")
    refine_yolo = YOLO(str(args.refine_weights), task="obb")
    audit_frame, failures = v3.audit_shared_checkpoint_state(ca_yolo.model, refine_yolo.model)
    audit_frame.to_csv(args.output_dir / "checkpoint_audit.csv", index=False)
    if failures:
        raise RuntimeError("CA/Refine 共享状态不一致，停止 Probe 审计")
    del ca_yolo

    core_model = refine_yolo.model.to(device).float().eval()
    for parameter in core_model.parameters():
        parameter.requires_grad_(False)
    refine_head = rd.find_refine_head(core_model)
    runtime_args = rd.read_refine_runtime_args(refine_head)
    actual_profile = str(runtime_args["refine_experiment"])
    if actual_profile != args.expect_refine_profile:
        raise RuntimeError(f"Refine profile={actual_profile!r}，预期 {args.expect_refine_profile!r}")
    cfg = rd.configure_model_args(core_model, args, runtime_args)
    criterion = core_model.init_criterion()
    rd.assert_refine_runtime_args(refine_head, runtime_args)
    data_dict = check_det_dataset(args.data)
    stride = max(int(core_model.stride.max().item()), 32)

    split_reservoirs = {}
    for split_index, split in enumerate((args.train_split, args.eval_split)):
        split_reservoirs[split] = {}
        for level_index, level in enumerate(("P3", "P4", "P5")):
            generator = torch.Generator().manual_seed(args.seed + split_index * 100 + level_index)
            split_reservoirs[split][level] = v3.PriorityReservoir(args.max_probe_samples_per_level, generator)

    train_loader = rd.build_split_loader(cfg, data_dict, args.train_split, args.batch, args.workers, stride)
    _, train_stats = v3.collect_split(
        split=args.train_split, loader=train_loader, core_model=core_model, criterion=criterion,
        refine_head=refine_head, rd=rd, make_anchors=make_anchors, args=args,
        reservoirs=split_reservoirs[args.train_split], keep_eval_rows=False,
    )
    del train_loader
    eval_loader = rd.build_split_loader(cfg, data_dict, args.eval_split, args.batch, args.workers, stride)
    eval_rows, eval_stats = v3.collect_split(
        split=args.eval_split, loader=eval_loader, core_model=core_model, criterion=criterion,
        refine_head=refine_head, rd=rd, make_anchors=make_anchors, args=args,
        reservoirs=split_reservoirs[args.eval_split], keep_eval_rows=True,
    )
    del eval_loader

    train_data = {level: reservoir.data for level, reservoir in split_reservoirs[args.train_split].items() if reservoir.data}
    eval_data = {level: reservoir.data for level, reservoir in split_reservoirs[args.eval_split].items() if reservoir.data}
    oracle_frame = v3.summarize_oracles(eval_rows, args.benefit_epsilon)
    metrics = run_probe_audit(train_data, eval_data, args=args, device=device)
    if metrics.empty:
        raise RuntimeError("没有任何 Probe 满足最小样本数，无法形成审计结果")
    candidates = summarize_candidates(metrics, oracle_frame, args)
    if candidates.empty:
        raise RuntimeError("未形成聚合候选，请检查 train/val 正样本与 scope")

    oracle_frame.to_csv(args.output_dir / "oracle_dof.csv", index=False)
    metrics.to_csv(args.output_dir / "probe_metrics.csv", index=False)
    candidates.to_csv(args.output_dir / "candidate_summary.csv", index=False)
    report_path = args.output_dir / "refine_v3_probe_audit.md"
    write_report(
        report_path, args=args, audit_frame=audit_frame, oracle_frame=oracle_frame,
        metrics=metrics, candidates=candidates, split_stats=(train_stats, eval_stats),
    )
    print("=" * 80)
    print(f"passing_candidates={int(candidates['pass'].sum())}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
