from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import os
os.environ["OMP_NUM_THREADS"] = "8"  # 修复 libgomp 警告

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.models.yolo.obb import OBBValidator

try:
    from myscripts2.tnms_topology_nms import TNMSConfig, topology_nms_single_image
except ModuleNotFoundError:
    from tnms_topology_nms import TNMSConfig, topology_nms_single_image



CORE_METRIC_KEYS = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "fitness",
]


def _clone_pred(pred: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """深拷贝单张图像预测字典，避免后续流程修改原对象。"""
    return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in pred.items()}


def _image_id_from_path(im_file: str) -> int | str:
    """从图像路径提取 image_id，兼容数字文件名。"""
    stem = Path(im_file).stem
    return int(stem) if stem.isnumeric() else stem


def _pred_to_numpy(pred: dict[str, torch.Tensor]) -> np.ndarray:
    """将预测字典转为 (N, 7) numpy: [x, y, w, h, theta, conf, cls]。"""
    if pred["cls"].numel() == 0:
        return np.zeros((0, 7), dtype=np.float32)

    arr = torch.cat([pred["bboxes"], pred["conf"].unsqueeze(-1), pred["cls"].unsqueeze(-1)], dim=1)
    return arr.detach().cpu().numpy().astype(np.float32)


def _numpy_to_pred(
    arr: np.ndarray,
    device: torch.device,
    box_dtype: torch.dtype,
    score_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """将 (N, 7) numpy 还原为验证器预测字典格式。"""
    if arr.size == 0:
        return {
            "bboxes": torch.zeros((0, 5), device=device, dtype=box_dtype),
            "conf": torch.zeros((0,), device=device, dtype=score_dtype),
            "cls": torch.zeros((0,), device=device, dtype=score_dtype),
        }

    t = torch.as_tensor(arr, device=device, dtype=score_dtype)
    return {
        "bboxes": t[:, :5].to(dtype=box_dtype),
        "conf": t[:, 5],
        "cls": t[:, 6],
    }


def _to_plain_dict(data: dict[str, Any]) -> dict[str, float]:
    """将指标字典中的 numpy/torch 标量转为 python float。"""
    out: dict[str, float] = {}
    for k, v in data.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif hasattr(v, "item"):
            out[k] = float(v.item())
        else:
            out[k] = float(v)
    return out


def _safe_summary(metrics_obj) -> list[dict[str, Any]]:
    """读取 per-class summary，失败时返回空列表。"""
    try:
        return metrics_obj.summary()
    except Exception:
        return []


class CollectingOBBValidator(OBBValidator):
    """在验证过程中收集并导出预测框（原图坐标系）。"""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)
        self.collected_predictions: list[dict[str, Any]] = []

    def _collect_one_image(self, pred: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """收集单张图像预测，写入内存列表。"""
        if pred["cls"].numel() == 0:
            return

        image_id = _image_id_from_path(pbatch["im_file"])
        file_name = Path(pbatch["im_file"]).name
        cls_tensor = pred["cls"].detach().cpu().numpy()
        conf_tensor = pred["conf"].detach().cpu().numpy()
        rbox_tensor = pred["bboxes"].detach().cpu().numpy()

        for r, s, c in zip(rbox_tensor.tolist(), conf_tensor.tolist(), cls_tensor.tolist()):
            c_int = int(c)
            cat_id = int(self.class_map[c_int]) if self.class_map is not None else c_int
            self.collected_predictions.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "category_id": cat_id,
                    "cls": c_int,
                    "score": float(s),
                    "rbox": [float(x) for x in r],
                }
            )

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """先收集预测，再走官方指标流程。"""
        for si, pred in enumerate(preds):
            pred_copy = _clone_pred(pred)
            predn = self._prepare_pred(pred_copy)
            if predn["cls"].numel() == 0:
                continue

            # 仅构造缩放预测所需的最小元信息，避免重复调用 _prepare_batch 修改 GT。
            pbatch = {
                "imgsz": batch["img"].shape[2:],
                "ori_shape": batch["ori_shape"][si],
                "ratio_pad": batch["ratio_pad"][si],
                "im_file": batch["im_file"][si],
            }
            pred_scaled = self.scale_preds(predn, pbatch)
            self._collect_one_image(pred_scaled, pbatch)

        super().update_metrics(preds, batch)


class TNMSOBBValidator(CollectingOBBValidator):
    """在官方 OBBValidator 后处理阶段插入 T-NMS。"""

    def __init__(self, tnms_cfg: TNMSConfig, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)
        self.tnms_cfg = tnms_cfg

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """先走官方 OBB 后处理，再执行 T-NMS 融合。"""
        preds = super().postprocess(preds)
        merged_preds: list[dict[str, torch.Tensor]] = []

        for pred in preds:
            arr = _pred_to_numpy(pred)
            merged = topology_nms_single_image(arr, self.tnms_cfg)
            merged_preds.append(
                _numpy_to_pred(
                    merged,
                    device=pred["bboxes"].device,
                    box_dtype=pred["bboxes"].dtype,
                    score_dtype=pred["conf"].dtype,
                )
            )

        return merged_preds


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "完整 T-NMS 评估流程：模型推理提取原始 OBB -> T-NMS 融合 -> "
            "计算并对比融合前后 mAP 等指标。"
        )
    )

    # 基础验证参数
    parser.add_argument("--model", type=Path, required=True, help="OBB 模型权重路径（.pt）。")
    parser.add_argument("--data", type=Path, required=True, help="数据集 yaml 路径。")
    parser.add_argument("--split", type=str, default="val", help="评估数据划分，默认 val。")
    parser.add_argument("--imgsz", type=int, default=640, help="验证图像尺寸。")
    parser.add_argument("--batch", type=int, default=8, help="验证 batch size。")
    parser.add_argument("--device", type=str, default="", help="设备，如 cpu / 0 / 0,1。")
    parser.add_argument("--workers", type=int, default=8, help="dataloader workers。")
    parser.add_argument("--conf", type=float, default=0.01, help="验证 conf 阈值。")
    parser.add_argument("--iou", type=float, default=0.7, help="验证 NMS IoU 阈值。")
    parser.add_argument("--max-det", type=int, default=300, help="每图最大保留框数。")
    parser.add_argument("--half", action="store_true", help="启用 FP16 验证。")
    parser.add_argument("--single-cls", action="store_true", help="单类别评估模式。")
    parser.add_argument("--agnostic-nms", action="store_true", help="类别无关 NMS。")

    # T-NMS 参数
    parser.add_argument("--angle-thr-deg", type=float, default=8.0, help="角度阈值（度）。")
    parser.add_argument("--perp-thr", type=float, default=8.0, help="法向共线阈值（像素）。")
    parser.add_argument("--gap-thr", type=float, default=20.0, help="轴向 gap 阈值（像素）。")
    parser.add_argument("--tnms-conf-thr", type=float, default=0.001, help="T-NMS 内部置信度过滤阈值。")
    parser.add_argument("--pre-nms-iou", type=float, default=0.0, help="T-NMS 前置官方旋转 NMS 的 IoU 阈值。")
    parser.add_argument("--class-agnostic-tnms", action="store_true", help="T-NMS 融合阶段启用类别无关。")

    # 输出参数
    parser.add_argument("--output-dir", type=Path, default=Path("tnms_results") / "tnms_eval_pipeline", help="输出目录。")

    return parser


def build_validator_args(cli_args: argparse.Namespace, run_name: str) -> dict[str, Any]:
    """组装官方验证器参数。"""
    return {
        "task": "obb",
        "mode": "val",
        "model": str(cli_args.model),
        "data": str(cli_args.data),
        "split": cli_args.split,
        "imgsz": int(cli_args.imgsz),
        "batch": int(cli_args.batch),
        "device": cli_args.device,
        "workers": int(cli_args.workers),
        "conf": float(cli_args.conf),
        "iou": float(cli_args.iou),
        "max_det": int(cli_args.max_det),
        "half": bool(cli_args.half),
        "single_cls": bool(cli_args.single_cls),
        "agnostic_nms": bool(cli_args.agnostic_nms),
        "save_json": False,
        "save_txt": False,
        "save_conf": False,
        "plots": False,
        "verbose": False,
        "project": str(cli_args.output_dir),
        "name": run_name,
        "exist_ok": True,
    }


def run_baseline_validation(model: YOLO, cli_args: argparse.Namespace) -> tuple[dict[str, float], CollectingOBBValidator]:
    """运行官方 OBB 验证（融合前）。"""
    validator = CollectingOBBValidator(args=build_validator_args(cli_args, run_name="before_tnms"))
    stats = validator(model=model.model)
    return _to_plain_dict(stats), validator


def run_tnms_validation(
    model: YOLO,
    cli_args: argparse.Namespace,
    tnms_cfg: TNMSConfig,
) -> tuple[dict[str, float], TNMSOBBValidator]:
    """运行插入 T-NMS 的 OBB 验证（融合后）。"""
    validator = TNMSOBBValidator(
        tnms_cfg=tnms_cfg,
        args=build_validator_args(cli_args, run_name="after_tnms"),
    )
    stats = validator(model=model.model)
    return _to_plain_dict(stats), validator


def build_compare_dict(before: dict[str, float], after: dict[str, float]) -> dict[str, dict[str, float]]:
    """构建核心指标前后对比。"""
    compare: dict[str, dict[str, float]] = {}
    for k in CORE_METRIC_KEYS:
        b = float(before.get(k, 0.0))
        a = float(after.get(k, 0.0))
        compare[k] = {"before": b, "after": a, "delta": a - b}
    return compare


class NumpyEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，支持 numpy 数据类型序列化。"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(path: Path, obj: Any) -> None:
    """写入 JSON 文件。"""
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, cls=NumpyEncoder), encoding="utf-8")


def render_markdown_report(
    before: dict[str, float],
    after: dict[str, float],
    compare: dict[str, dict[str, float]],
    n_before: int,
    n_after: int,
    n_images_before: int,
    n_images_after: int,
    cli_args: argparse.Namespace,
    tnms_cfg: TNMSConfig,
) -> str:
    """生成对比分析 Markdown 报告。"""

    def f(x: float) -> str:
        return f"{x:.6f}"

    lines = [
        "# T-NMS 融合前后指标对比报告",
        "",
        "## 1. 实验配置",
        "",
        f"- 模型：`{cli_args.model}`",
        f"- 数据：`{cli_args.data}`（split={cli_args.split}）",
        f"- 验证参数：imgsz={cli_args.imgsz}, batch={cli_args.batch}, conf={cli_args.conf}, iou={cli_args.iou}, max_det={cli_args.max_det}",
        (
            "- T-NMS 参数："
            f"angle_thr_deg={tnms_cfg.angle_thr_deg}, perp_thr={tnms_cfg.perp_thr}, gap_thr={tnms_cfg.gap_thr}, "
            f"conf_thr={tnms_cfg.conf_thr}, pre_nms_iou={tnms_cfg.pre_nms_iou}, class_aware={tnms_cfg.class_aware}"
        ),
        "",
        "## 2. 核心指标对比",
        "",
        "| 指标 | 融合前 | 融合后 | 变化(后-前) |",
        "|---|---:|---:|---:|",
    ]

    for k in CORE_METRIC_KEYS:
        row = compare[k]
        lines.append(f"| {k} | {f(row['before'])} | {f(row['after'])} | {f(row['delta'])} |")

    lines.extend(
        [
            "",
            "## 3. 预测框数量变化",
            "",
            "| 统计项 | 融合前 | 融合后 | 变化(后-前) |",
            "|---|---:|---:|---:|",
            f"| 总预测框数 | {n_before} | {n_after} | {n_after - n_before} |",
            f"| 图像数 | {n_images_before} | {n_images_after} | {n_images_after - n_images_before} |",
            (
                f"| 平均每图预测框 | {n_before / max(n_images_before, 1):.6f} | "
                f"{n_after / max(n_images_after, 1):.6f} | "
                f"{(n_after / max(n_images_after, 1)) - (n_before / max(n_images_before, 1)):.6f} |"
            ),
            "",
            "## 4. 结论建议",
            "",
            "- 若 `metrics/mAP50-95(B)` 和 `metrics/mAP50(B)` 同步上升，说明 T-NMS 对细长目标断裂预测有正向修复作用。",
            "- 若 Recall 提升但 Precision 明显下降，建议收紧 `gap_thr` 或 `perp_thr`。",
            "- 若提升不明显，可尝试：减小 `angle_thr_deg`、开启 `pre_nms_iou`（如 0.2~0.4）再对比。",
            "",
            "## 5. 指标原始字典",
            "",
            "### 融合前",
            "```json",
            json.dumps(before, ensure_ascii=False, indent=2),
            "```",
            "",
            "### 融合后",
            "```json",
            json.dumps(after, ensure_ascii=False, indent=2),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    """主流程：模型推理提取预测 -> T-NMS -> 融合前后指标对比。"""
    args = build_parser().parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"模型文件不存在: {args.model}")
    if not args.data.exists():
        raise FileNotFoundError(f"数据配置不存在: {args.data}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tnms_cfg = TNMSConfig(
        angle_thr_deg=float(args.angle_thr_deg),
        perp_thr=float(args.perp_thr),
        gap_thr=float(args.gap_thr),
        conf_thr=float(args.tnms_conf_thr),
        pre_nms_iou=float(args.pre_nms_iou),
        class_aware=not bool(args.class_agnostic_tnms),
    )

    model = YOLO(str(args.model))
    if model.task != "obb":
        raise ValueError(f"当前模型任务为 {model.task}，该脚本仅支持 OBB 模型。")

    before_stats, before_validator = run_baseline_validation(model, args)
    after_stats, after_validator = run_tnms_validation(model, args, tnms_cfg)

    compare = build_compare_dict(before_stats, after_stats)

    before_records = before_validator.collected_predictions
    after_records = after_validator.collected_predictions

    outputs = {
        "predictions_before": output_dir / "predictions_before_tnms.json",
        "predictions_after": output_dir / "predictions_after_tnms.json",
        "metrics_before": output_dir / "metrics_before_tnms.json",
        "metrics_after": output_dir / "metrics_after_tnms.json",
        "metrics_compare": output_dir / "metrics_compare.json",
        "class_metrics_before": output_dir / "class_metrics_before_tnms.json",
        "class_metrics_after": output_dir / "class_metrics_after_tnms.json",
        "tnms_config": output_dir / "tnms_config.json",
        "report": output_dir / "compare_report.md",
    }

    save_json(outputs["predictions_before"], before_records)
    save_json(outputs["predictions_after"], after_records)
    save_json(outputs["metrics_before"], before_stats)
    save_json(outputs["metrics_after"], after_stats)
    save_json(outputs["metrics_compare"], compare)
    save_json(outputs["class_metrics_before"], _safe_summary(before_validator.metrics))
    save_json(outputs["class_metrics_after"], _safe_summary(after_validator.metrics))
    save_json(outputs["tnms_config"], {
        "angle_thr_deg": tnms_cfg.angle_thr_deg,
        "perp_thr": tnms_cfg.perp_thr,
        "gap_thr": tnms_cfg.gap_thr,
        "conf_thr": tnms_cfg.conf_thr,
        "pre_nms_iou": tnms_cfg.pre_nms_iou,
        "class_aware": tnms_cfg.class_aware,
    })

    report_md = render_markdown_report(
        before=before_stats,
        after=after_stats,
        compare=compare,
        n_before=len(before_records),
        n_after=len(after_records),
        n_images_before=len({_r["image_id"] for _r in before_records}),
        n_images_after=len({_r["image_id"] for _r in after_records}),
        cli_args=args,
        tnms_cfg=tnms_cfg,
    )
    outputs["report"].write_text(report_md, encoding="utf-8")

    print("T-NMS 对比评估完成。")
    print(f"融合前 mAP50-95: {before_stats.get('metrics/mAP50-95(B)', 0.0):.6f}")
    print(f"融合后 mAP50-95: {after_stats.get('metrics/mAP50-95(B)', 0.0):.6f}")
    print(
        "mAP50-95 变化: "
        f"{after_stats.get('metrics/mAP50-95(B)', 0.0) - before_stats.get('metrics/mAP50-95(B)', 0.0):+.6f}"
    )
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
