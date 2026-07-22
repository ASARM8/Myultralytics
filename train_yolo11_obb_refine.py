"""
YOLOv11-OBB + Refine Head 训练脚本
基于 CA (Coverage-Aware) 模型，新增轻量宽高精修分支（Δw, Δh）。
使用方法:
    直接运行: python train_yolo11_obb_refine.py
    修改下方 CONFIG 字典中的参数即可自定义训练配置

关键设计：
    - 从 yolo11-obb-ca-refine.yaml 构建模型（含 OBBRefine 头 + cv5 精修分支）
    - 不加载任何 .pt 预训练权重，所有参数从 YAML 随机初始化开始训练
    - aux_geo 设为 0.2（作为 refine residual 宽高监督增益）
"""

import argparse
import csv
import os
import shutil
import traceback
import datetime
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "8"

import torch
from ultralytics import YOLO
from ultralytics.utils.logger import ConsoleLogger
from ultralytics.utils.torch_utils import unwrap_model

# ========================== 路径配置 ==========================
# 不使用预训练权重，保留该变量仅用于打印和 A/B 默认路径逻辑
PRETRAIN_WEIGHTS = None

# OBBRefine YAML 配置路径（含 cv5 精修分支定义）
MODEL_YAML = "ultralytics/cfg/models/11/yolo11l-obb-ca-refine.yaml"
RUN_NAME = "yolo11_obb_640_811_ca_refine_scratch"

VAL_WEIGHTS = f"/root/autodl-tmp/work-dirs/{RUN_NAME}/weights/best.pt"
RESUME_WEIGHTS = f"/root/autodl-tmp/work-dirs/{RUN_NAME}/weights/epoch290.pt"
RESUME_TOTAL_EPOCHS = 400
RESUME_LR0 = 5e-6
RESUME_LRF = 1.0
RESUME_WARMUP_EPOCHS = 0.0

# ========================== 训练配置 ==========================
CONFIG = {
    # ---------- 模型配置 ----------
    # 从 YAML 构建新架构（不从 .pt 加载，完全随机初始化）
    "model": MODEL_YAML,

    # ---------- 数据集配置 ----------
    "data": "/root/autodl-tmp/datasets/TTPLA-640-811/dataset.yaml",

    # ---------- 训练基本参数 ----------
    "epochs": 300,
    "batch": 16,
    "imgsz": 640,
    "device": 0,
    "workers": 16,  # 临时设为 0 排查多进程崩溃根因；确认无误后改回 8
    "patience": 0,

    # ---------- 输出目录配置 ----------
    "project": "/root/autodl-tmp/work-dirs",
    "name": RUN_NAME,
    "exist_ok": False,

    # ---------- 模型保存配置 ----------
    "save": True,
    "save_period": 10,

    # ---------- 验证与可视化 ----------
    "val": True,
    "plots": True,

    # ---------- 训练策略 ----------
    "pretrained": False,
    "optimizer": "AdamW",
    "lr0": 0.0003,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 10.0,
    "cos_lr": True,
    "close_mosaic": 100,

    # ---------- 数据增强配置 ----------
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.05,
    "scale": 0.3,
    "fliplr": 0.5,
    "flipud": 0.0,
    "mosaic": 0.5,
    "mixup": 0.0,

    # ---------- 损失权重 ----------
    # aux_geo: 辅助几何损失增益（梯度现在通过 cv5 连续参数有效流回）
    "aux_geo": 0.2,
    "aux_geo_lp": 0.0,    # L_perp（法向偏移）：本次不做 Δn，关闭
    "aux_geo_lw": 2.0,
    "aux_geo_lt": 0.0,
    "aux_geo_ar": 30.0,             
    "aux_geo_ws": 16.0,
    "refine_feature_detach": True,

    # ---------- 其他 ----------
    "amp": False,
    "cache": 'disk',
    "resume": False,
    "seed": 0,
    "verbose": True,
}


def move_log_to_save_dir(log_file, save_dir):
    log_file = Path(log_file)
    if not save_dir:
        return log_file

    save_dir = Path(save_dir)
    if not save_dir.exists():
        return log_file

    if log_file.exists() and log_file.parent.resolve() == save_dir.resolve():
        return log_file

    target_file = save_dir / "train_console.log"
    if log_file.resolve() == target_file.resolve():
        return target_file

    if not log_file.exists():
        return target_file if target_file.exists() else log_file

    if target_file.exists():
        stem, suffix = target_file.stem, target_file.suffix
        index = 2
        while (save_dir / f"{stem}_{index}{suffix}").exists():
            index += 1
        target_file = save_dir / f"{stem}_{index}{suffix}"

    shutil.move(str(log_file), str(target_file))
    return target_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "resume", "val_ab"], default="train")
    parser.add_argument("--weights", default=VAL_WEIGHTS)
    parser.add_argument("--resume-weights", default=RESUME_WEIGHTS)
    parser.add_argument("--total-epochs", type=int, default=RESUME_TOTAL_EPOCHS)
    parser.add_argument("--resume-lr0", type=float, default=RESUME_LR0)
    parser.add_argument("--resume-lrf", type=float, default=RESUME_LRF)
    parser.add_argument("--resume-warmup-epochs", type=float, default=RESUME_WARMUP_EPOCHS)
    parser.add_argument("--resume-cos-lr", action="store_true")
    parser.add_argument("--resume-name", default=None)
    parser.add_argument("--data", default=CONFIG["data"])
    parser.add_argument("--imgsz", type=int, default=CONFIG["imgsz"])
    parser.add_argument("--batch", type=int, default=CONFIG["batch"])
    parser.add_argument("--device", default=str(CONFIG["device"]))
    return parser.parse_args()


def load_torch_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def prepare_stable_resume_checkpoint(args) -> Path:
    source = Path(args.resume_weights)
    if not source.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {source}")

    checkpoint = load_torch_checkpoint(source)
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    start_epoch = checkpoint_epoch + 1
    if args.total_epochs <= start_epoch:
        raise ValueError(
            f"total_epochs must be greater than checkpoint start epoch {start_epoch}; got {args.total_epochs}."
        )

    train_args = checkpoint.setdefault("train_args", {})
    original_epochs = train_args.get("epochs")
    original_lr0 = train_args.get("lr0")
    original_lrf = train_args.get("lrf")
    original_cos_lr = train_args.get("cos_lr")

    train_args["epochs"] = int(args.total_epochs)
    train_args["lr0"] = float(args.resume_lr0)
    train_args["lrf"] = float(args.resume_lrf)
    train_args["cos_lr"] = bool(args.resume_cos_lr)
    train_args["warmup_epochs"] = float(args.resume_warmup_epochs)
    train_args["close_mosaic"] = int(args.total_epochs)
    train_args["project"] = CONFIG["project"]
    train_args["name"] = args.resume_name or f"{RUN_NAME}_stable_resume_epoch{checkpoint_epoch}_to_{args.total_epochs}"
    train_args["exist_ok"] = False
    train_args.pop("save_dir", None)

    target = source.with_name(f"{source.stem}_stable_resume_to_{args.total_epochs}{source.suffix}")
    torch.save(checkpoint, target)

    print(
        f"[*] Created stable resume checkpoint: {target}\n"
        f"    source_epoch={checkpoint_epoch}, original_epochs={original_epochs}, total_epochs={args.total_epochs}\n"
        f"    lr0: {original_lr0} -> {args.resume_lr0}, lrf: {original_lrf} -> {args.resume_lrf}, "
        f"cos_lr: {original_cos_lr} -> {bool(args.resume_cos_lr)}\n"
        f"    warmup_epochs={args.resume_warmup_epochs}, close_mosaic={args.total_epochs}\n"
        f"    output_name={train_args['name']}"
    )
    if checkpoint.get("optimizer") is None:
        print("[!] Warning: checkpoint has no optimizer state; this will behave like fine-tuning, not strict resume.")
    return target


def set_refine_inference_mode(yolo_model, disable_refine_inference: bool):
    from ultralytics.nn.modules.head import OBBRefine

    model_root = yolo_model.model if hasattr(yolo_model, "model") else yolo_model
    found = 0
    for m in model_root.modules():
        if isinstance(m, OBBRefine):
            m.disable_refine_inference = bool(disable_refine_inference)
            found += 1
    if found == 0:
        raise RuntimeError("未找到 OBBRefine 模块，无法切换 coarse-only 验证模式。")


def sync_obbrefine_runtime_attrs(
    target_model,
    refine_select_ar: float,
    refine_select_ws: float,
    disable_refine_inference: bool,
    refine_feature_detach: bool,
):
    from ultralytics.nn.modules.head import OBBRefine

    model_root = target_model.model if hasattr(target_model, "model") else target_model
    found = 0
    for m in model_root.modules():
        if isinstance(m, OBBRefine):
            m.refine_select_ar = float(refine_select_ar)
            m.refine_select_ws = float(refine_select_ws)
            m.disable_refine_inference = bool(disable_refine_inference)
            m.refine_feature_detach = bool(refine_feature_detach)
            found += 1
    if found == 0:
        raise RuntimeError("未找到 OBBRefine 模块，无法同步运行时属性。")


def run_val_ab(args):
    log_file = Path(CONFIG["project"]) / f"{CONFIG['name']}_val_ab_console.tmp.log"
    console_logger = ConsoleLogger(log_file, batch_size=1)
    console_logger.start_capture()
    final_log_file = log_file
    model = YOLO(args.weights)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    common_kwargs = {
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": CONFIG["project"],
        "exist_ok": True,
        "plots": True,
    }
    metric_keys = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    metric_labels = {
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
        "metrics/mAP50(B)": "mAP50",
        "metrics/mAP50-95(B)": "mAP50-95",
    }

    print("=" * 60)
    print(f"  A/B 验证权重: {args.weights}")
    print(f"  数据集: {args.data}")
    print(f"  图片尺寸: {args.imgsz}")
    print(f"  批次大小: {args.batch}")
    print("=" * 60)

    try:
        set_refine_inference_mode(model, False)
        normal_metrics = model.val(name=f"{CONFIG['name']}-val-normal-{stamp}", **common_kwargs)
        normal_dict = normal_metrics.results_dict
        final_log_file = move_log_to_save_dir(final_log_file, normal_metrics.save_dir)
        console_logger.destination = final_log_file

        set_refine_inference_mode(model, True)
        coarse_metrics = model.val(name=f"{CONFIG['name']}-val-coarse-only-{stamp}", **common_kwargs)
        coarse_dict = coarse_metrics.results_dict

        print("\n" + "=" * 60)
        print("  A/B 验证结果对比")
        print("=" * 60)
        for key in metric_keys:
            normal_value = float(normal_dict[key])
            coarse_value = float(coarse_dict[key])
            delta = coarse_value - normal_value
            print(
                f"  {metric_labels[key]}: normal={normal_value:.5f}, coarse-only={coarse_value:.5f}, Δ={delta:+.5f}"
            )
        print(f"  日志文件: {final_log_file}")
        print("=" * 60)

        return {"normal": normal_dict, "coarse_only": coarse_dict}
    except Exception:
        traceback.print_exc()
        raise
    finally:
        console_logger.stop_capture()


def main(model_path: str | Path | None = None, train_overrides: dict | None = None):
    """主训练函数"""
    log_file = Path(CONFIG["project"]) / f"{CONFIG['name']}_train_console.tmp.log"
    console_logger = ConsoleLogger(log_file, batch_size=1)
    console_logger.start_capture()
    model = None
    final_log_file = log_file

    def attach_log_to_save_dir(trainer):
        nonlocal final_log_file
        console_logger._flush_buffer()
        final_log_file = move_log_to_save_dir(final_log_file, trainer.save_dir)
        console_logger.destination = final_log_file

    try:
        print("\n[*] 终端日志拦截器已启动\n")

        # 1. 从 YAML 构建模型（含 OBBRefine 头）
        model = YOLO(str(model_path) if model_path else CONFIG["model"])
        model.add_callback("on_pretrain_routine_start", attach_log_to_save_dir)

        # 2. 注册回调：训练开始前同步 OBBRefine 运行时属性
        def on_pretrain_routine_end(trainer):
            """在模型构建完成后、训练循环开始前，同步 OBBRefine 运行时属性。"""
            sync_obbrefine_runtime_attrs(
                trainer.model,
                trainer.args.aux_geo_ar,
                trainer.args.aux_geo_ws,
                True,
                trainer.args.refine_feature_detach,
            )
            if hasattr(trainer, 'ema') and trainer.ema is not None:
                sync_obbrefine_runtime_attrs(
                    trainer.ema.ema,
                    trainer.args.aux_geo_ar,
                    trainer.args.aux_geo_ws,
                    True,
                    trainer.args.refine_feature_detach,
                )
            print(
                f"[*] OBBRefine 运行时属性已同步: AR>{float(trainer.args.aux_geo_ar)}, "
                f"short<{float(trainer.args.aux_geo_ws)}px, "
                f"refine_feature_detach={bool(trainer.args.refine_feature_detach)}, "
                f"默认验证/推理=coarse-only，训练从 YAML 随机初始化开始"
            )

        model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)

        # 3. 注册回调：日志文件落地
        def on_train_start(trainer):
            attach_log_to_save_dir(trainer)
            print(f"\n[*] 日志文件: {final_log_file}\n")
            trainer.refine_diag_file = os.path.join(trainer.save_dir, "refine_diag.csv")
            with open(trainer.refine_diag_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "refine_mask_ratio", "avg_abs_dshort", "refine_loss"])

        model.add_callback("on_train_start", on_train_start)

        def on_train_epoch_end(trainer):
            criterion = getattr(unwrap_model(trainer.model), "criterion", None)
            if criterion is None or not hasattr(criterion, "refine_diagnostics"):
                return
            diag = criterion.refine_diagnostics
            print(
                f"[*] refine_diag epoch {trainer.epoch + 1}: "
                f"mask_ratio={float(diag['refine_mask_ratio']):.6f}, "
                f"avg_abs_dshort={float(diag['avg_abs_dshort']):.6f}, "
                f"refine_loss={float(diag['refine_loss']):.6f}"
            )
            diag_file = getattr(trainer, "refine_diag_file", None)
            if diag_file:
                with open(diag_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        trainer.epoch + 1,
                        float(diag["refine_mask_ratio"]),
                        float(diag["avg_abs_dshort"]),
                        float(diag["refine_loss"]),
                    ])

        model.add_callback("on_train_epoch_end", on_train_epoch_end)

        # 4. 打印配置
        if not CONFIG["refine_feature_detach"]:
            raise ValueError("最终保留路线要求 refine_feature_detach=True。")
        refine_mode_label = "fully-decoupled"

        print("=" * 60)
        print(f"  模型 YAML: {MODEL_YAML}")
        print("  预训练权重: 不使用（从 YAML 随机初始化）")
        print(f"  数据集: {CONFIG['data']}")
        print(f"  训练轮数: {CONFIG['epochs']}")
        print(f"  批次大小: {CONFIG['batch']}")
        print(f"  图片尺寸: {CONFIG['imgsz']}")
        print(f"  输出目录: {CONFIG['project']}/{CONFIG['name']}")
        print(f"  临时日志文件: {log_file}")
        print(f"  aux_geo 增益: {CONFIG['aux_geo']}")
        print(f"  Refine Head: Δw + Δh (ne_refine=2, {refine_mode_label})")
        print(f"  refine_feature_detach: {CONFIG['refine_feature_detach']}")
        print("  验证/推理默认口径: coarse-only（默认禁用 refine inference）")
        print("  A/B 对照入口: python train_yolo11_obb_refine.py --mode val_ab --weights <ckpt>")
        print("=" * 60)

        # 5. 开始训练
        train_config = CONFIG.copy()
        if train_overrides:
            train_config.update(train_overrides)
        results = model.train(**train_config)
        save_dir = getattr(model.trainer, "save_dir", Path(CONFIG["project"]) / CONFIG["name"])

        # 6. 训练完成
        print("\n" + "=" * 60)
        print("  训练完成！")
        print(f"  结果保存在: {save_dir}")
        print(f"  日志文件: {final_log_file}")
        print("  保存内容说明:")
        print("    - weights/best.pt     : 按 coarse-only 验证指标选出的最佳模型")
        print("    - weights/last.pt     : 最后一轮权重")
        print("    - results.csv         : 每轮训练指标（默认 coarse-only 验证口径）")
        print("    - refine_diag.csv     : 每轮 refine 诊断统计")
        print("    - results.png         : 训练曲线图")
        print("    - args.yaml           : 完整训练参数记录")
        print("=" * 60)

        return results
    except Exception:
        traceback.print_exc()
        raise
    finally:
        console_logger.stop_capture()
        trainer = getattr(model, "trainer", None) if model is not None else None
        save_dir = getattr(trainer, "save_dir", None)
        final_log_file = move_log_to_save_dir(final_log_file, save_dir)
        print(f"  日志文件: {final_log_file}")


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.mode == "val_ab":
        run_val_ab(cli_args)
    elif cli_args.mode == "resume":
        stable_resume_weights = prepare_stable_resume_checkpoint(cli_args)
        main(
            model_path=stable_resume_weights,
            train_overrides={
                "resume": True,
                "close_mosaic": int(cli_args.total_epochs),
                "save_period": CONFIG["save_period"],
                "workers": CONFIG["workers"],
                "batch": cli_args.batch,
                "device": cli_args.device,
                "imgsz": cli_args.imgsz,
            },
        )
    else:
        main()
