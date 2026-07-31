"""
YOLOv11-OBB + Refine Head 训练脚本
基于 CA (Coverage-Aware) 模型，新增轻量宽高精修分支（Δw, Δh）。
使用方法:
    直接运行: python train_yolo11_obb_refine.py
    V2仅训练Refine: python train_yolo11_obb_refine.py --mode refine_only --base-weights <ca.pt>
    V2.1保守精修: 在上一命令后增加 --refine-experiment conservative_short_long
    修改下方 CONFIG 字典中的参数即可自定义训练配置

关键设计：
    - 从 yolo11-obb-ca-refine.yaml 构建模型（含 OBBRefine 头 + cv5 精修分支）
    - 不加载任何 .pt 预训练权重，所有参数从 YAML 随机初始化开始训练
    - aux_geo 设为 0.2（作为 refine residual 宽高监督增益）
    - refine_only 从纯CA权重初始化V2，冻结全部共享参数并只训练cv5
"""

import argparse
import copy
import csv
import datetime
import hashlib
import os
import shutil
import traceback
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
MODEL_V2_YAML = "ultralytics/cfg/models/11/yolo11l-obb-ca-refine-v2.yaml"
RUN_NAME = "yolo11_obb_640_811_ca_refine_scratch"
REFINE_EXPERIMENT_CONFIGS = {
    "bounded_wh": {"run_version": "v2", "refine_delta_max": 0.1, "refine_target_limit": 0.095},
    "direct_short_long": {"run_version": "v2", "refine_delta_max": 0.1, "refine_target_limit": 0.095},
    "aligned_gate": {"run_version": "v2", "refine_delta_max": 0.1, "refine_target_limit": 0.095},
    "aligned_identity": {"run_version": "v2", "refine_delta_max": 0.1, "refine_target_limit": 0.095},
    # V2.1 keeps the R2 topology/loss/gate and only reduces the trainable residual range.
    "conservative_short_long": {"run_version": "v21", "refine_delta_max": 0.05, "refine_target_limit": 0.04},
}
REFINE_EXPERIMENTS = tuple(REFINE_EXPERIMENT_CONFIGS)

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
    "refine_experiment": "legacy",
    "refine_delta_max": 0.1,
    "refine_target_limit": 0.095,
    "refine_smooth_l1_beta": 0.02,
    "refine_identity_gain": 0.05,

    # ---------- 其他 ----------
    "amp": False,
    "cache": 'disk',
    "resume": False,
    "seed": 0,
    "verbose": True,
}


def get_refine_experiment_config(experiment: str) -> dict[str, str | float]:
    """Return an isolated profile configuration without changing legacy V2 defaults."""
    try:
        return dict(REFINE_EXPERIMENT_CONFIGS[experiment])
    except KeyError as error:
        raise ValueError(f"Unknown Refine experiment {experiment!r}; expected one of {REFINE_EXPERIMENTS}") from error


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
    parser.add_argument("--mode", choices=["train", "refine_only", "resume", "val_ab"], default="train")
    parser.add_argument("--weights", default=VAL_WEIGHTS)
    parser.add_argument("--base-weights", help="refine_only 模式使用的纯 CA checkpoint")
    parser.add_argument("--refine-experiment", choices=REFINE_EXPERIMENTS, default="aligned_identity")
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
    parser.add_argument("--workers", type=int, default=CONFIG["workers"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--project", default=CONFIG["project"])
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    if args.imgsz != 640:
        parser.error("创新点一 Refine 实验固定使用 imgsz=640")
    if args.mode == "refine_only" and not args.base_weights:
        parser.error("--mode refine_only 必须提供 --base-weights <CA checkpoint>")
    return args


def load_torch_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def shared_state_digest(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash all non-Refine tensors to verify that trainer setup preserves the CA base."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        if ".cv5." in key or ".one2one_cv5." in key or key.endswith("._refine_v2_marker"):
            continue
        value = state_dict[key].detach().cpu().contiguous().reshape(-1)
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(state_dict[key].shape)).encode("ascii"))
        digest.update(str(state_dict[key].dtype).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_ca_weights_into_refine_v2(yolo_model, weights: str | Path) -> dict:
    """Load a pure CA checkpoint and require every non-cv5 tensor to match exactly."""
    from ultralytics.nn.modules.head import OBBRefineV2
    from ultralytics.nn.tasks import OBBModel

    source = Path(weights)
    if not source.exists():
        raise FileNotFoundError(f"CA checkpoint not found: {source}")
    checkpoint = load_torch_checkpoint(source)
    source_model = checkpoint.get("ema") or checkpoint.get("model")
    if source_model is None or not hasattr(source_model, "state_dict"):
        raise ValueError(f"checkpoint 未包含可加载的 model/ema: {source}")
    source_model = source_model.float()
    source_head = source_model.model[-1]
    source_nc = int(source_head.nc)
    target_head = yolo_model.model.model[-1]
    if int(target_head.nc) != source_nc:
        target_yaml = copy.deepcopy(yolo_model.model.yaml)
        source_channels = int((getattr(source_model, "yaml", {}) or {}).get("channels", 3))
        yolo_model.model = OBBModel(
            target_yaml,
            ch=source_channels,
            nc=source_nc,
            verbose=True,
        )

    source_state = source_model.state_dict()
    if any(".cv5." in key or ".one2one_cv5." in key for key in source_state):
        raise ValueError("--base-weights 必须是纯 CA checkpoint，不能使用 legacy/V2 Refine checkpoint")

    target_model = yolo_model.model
    heads = [module for module in target_model.modules() if isinstance(module, OBBRefineV2)]
    if len(heads) != 1:
        raise RuntimeError(f"V2 YAML 应只包含一个 OBBRefineV2，实际找到 {len(heads)} 个")

    target_state = target_model.state_dict()
    shared_state = {}
    missing_shared = []
    shape_mismatches = []
    expected_cv5 = []
    for key, target_value in target_state.items():
        if ".cv5." in key or ".one2one_cv5." in key or key.endswith("._refine_v2_marker"):
            expected_cv5.append(key)
            continue
        source_value = source_state.get(key)
        if source_value is None:
            missing_shared.append(key)
        elif source_value.shape != target_value.shape:
            shape_mismatches.append((key, tuple(source_value.shape), tuple(target_value.shape)))
        else:
            shared_state[key] = source_value
    if missing_shared or shape_mismatches:
        raise RuntimeError(
            "CA→RefineV2 共享参数不完整："
            f"missing={missing_shared[:8]}, shape_mismatches={shape_mismatches[:8]}"
        )
    if not expected_cv5:
        raise RuntimeError("V2 模型中未找到 cv5 参数，拒绝继续")

    incompatible = target_model.load_state_dict(shared_state, strict=False)
    unexpected_missing = [
        key
        for key in incompatible.missing_keys
        if ".cv5." not in key and ".one2one_cv5." not in key and not key.endswith("._refine_v2_marker")
    ]
    if incompatible.unexpected_keys or unexpected_missing:
        raise RuntimeError(
            f"CA→RefineV2 load_state_dict 不符合预期: missing={unexpected_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    loaded_state = target_model.state_dict()
    unequal = [
        key
        for key, source_value in shared_state.items()
        if not torch.equal(loaded_state[key].cpu(), source_value.to(dtype=loaded_state[key].dtype).cpu())
    ]
    if unequal:
        raise RuntimeError(f"共享 CA 参数加载后未保持逐元素一致: {unequal[:8]}")

    # Model.train() rebuilds the task model unless ``ckpt`` is truthy. Mark this
    # in-memory V2 model as pretrained so the exact CA transfer survives trainer setup.
    yolo_model.ckpt = {"refine_v2_ca_init": True, "source": str(source)}
    return {
        "source": str(source),
        "nc": source_nc,
        "shared_tensors": len(shared_state),
        "new_v2_tensors": len(expected_cv5),
        "shared_digest": shared_state_digest(target_model.state_dict()),
    }


def configure_refine_only_training(trainer) -> dict:
    """Freeze every parameter except V2 cv5 and prune frozen tensors from the optimizer."""
    from ultralytics.nn.modules.head import OBBRefineV2

    core_model = unwrap_model(trainer.model)
    heads = [module for module in core_model.modules() if isinstance(module, OBBRefineV2)]
    if len(heads) != 1:
        raise RuntimeError(f"refine_only 需要且仅允许一个 OBBRefineV2，实际找到 {len(heads)} 个")
    head = heads[0]

    trainable_ids = {id(parameter) for parameter in head.cv5.parameters()}
    for parameter in core_model.parameters():
        parameter.requires_grad_(id(parameter) in trainable_ids)
    trainable = [parameter for parameter in core_model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("refine_only 冻结后没有可训练参数")

    for group in trainer.optimizer.param_groups:
        group["params"] = [parameter for parameter in group["params"] if parameter.requires_grad]
    trainer.optimizer.param_groups[:] = [group for group in trainer.optimizer.param_groups if group["params"]]
    if not trainer.optimizer.param_groups:
        raise RuntimeError("refine_only 优化器中没有保留 cv5 参数")
    for parameter in list(trainer.optimizer.state):
        if not parameter.requires_grad:
            del trainer.optimizer.state[parameter]

    return {
        "trainable_tensors": len(trainable),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "frozen_parameters": sum(
            parameter.numel() for parameter in core_model.parameters() if not parameter.requires_grad
        ),
    }


def keep_frozen_batch_norm_eval(trainer) -> None:
    """Prevent non-cv5 BatchNorm running statistics from changing in refine-only training."""
    from torch.nn.modules.batchnorm import _BatchNorm

    core_model = unwrap_model(trainer.model)
    for name, module in core_model.named_modules():
        if isinstance(module, _BatchNorm) and ".cv5." not in f".{name}.":
            module.eval()


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
    refine_experiment: str = "aligned_identity",
    refine_delta_max: float = 0.1,
    refine_target_limit: float = 0.095,
    refine_smooth_l1_beta: float = 0.02,
    refine_identity_gain: float = 0.05,
):
    from ultralytics.nn.modules.head import OBBRefine, OBBRefineV2

    model_root = target_model.model if hasattr(target_model, "model") else target_model
    found = 0
    for m in model_root.modules():
        if isinstance(m, OBBRefine):
            m.refine_select_ar = float(refine_select_ar)
            m.refine_select_ws = float(refine_select_ws)
            m.disable_refine_inference = bool(disable_refine_inference)
            m.refine_feature_detach = bool(refine_feature_detach)
            if isinstance(m, OBBRefineV2):
                if refine_delta_max <= 0:
                    raise ValueError("refine_delta_max 必须大于 0")
                if not 0 < refine_target_limit < refine_delta_max:
                    raise ValueError("refine_target_limit 必须位于 (0, refine_delta_max)")
                if refine_smooth_l1_beta <= 0 or refine_identity_gain < 0:
                    raise ValueError("refine_smooth_l1_beta 必须大于 0，refine_identity_gain 不能为负")
                m.set_refine_experiment(refine_experiment)
                m.refine_delta_max = float(refine_delta_max)
                m.refine_target_limit = float(refine_target_limit)
                m.refine_smooth_l1_beta = float(refine_smooth_l1_beta)
                m.refine_identity_gain = float(refine_identity_gain)
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
            delta = normal_value - coarse_value
            print(
                f"  {metric_labels[key]}: normal={normal_value:.5f}, coarse-only={coarse_value:.5f}, "
                f"Δ(normal-coarse)={delta:+.5f}"
            )
        print(f"  日志文件: {final_log_file}")
        print("=" * 60)

        return {"normal": normal_dict, "coarse_only": coarse_dict}
    except Exception:
        traceback.print_exc()
        raise
    finally:
        console_logger.stop_capture()


def main(
    model_path: str | Path | None = None,
    train_overrides: dict | None = None,
    *,
    refine_only: bool = False,
    base_weights: str | Path | None = None,
):
    """主训练函数"""
    train_config = CONFIG.copy()
    if train_overrides:
        train_config.update(train_overrides)
    log_file = Path(train_config["project"]) / f"{train_config['name']}_train_console.tmp.log"
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
        model = YOLO(str(model_path) if model_path else train_config["model"])
        load_summary = None
        if refine_only:
            if not base_weights:
                raise ValueError("refine_only=True 时必须提供 base_weights")
            load_summary = load_ca_weights_into_refine_v2(model, base_weights)
            print(
                f"[*] 已加载纯 CA 权重: {load_summary['source']}\n"
                f"    nc: {load_summary['nc']}, 共享张量: {load_summary['shared_tensors']}, "
                f"新 V2/cv5 张量: {load_summary['new_v2_tensors']}"
            )
        model.add_callback("on_pretrain_routine_start", attach_log_to_save_dir)

        # 2. 注册回调：训练开始前同步 OBBRefine 运行时属性
        def on_pretrain_routine_end(trainer):
            """在模型构建完成后、训练循环开始前，同步 OBBRefine 运行时属性。"""
            if refine_only and int(unwrap_model(trainer.model).model[-1].nc) != int(load_summary["nc"]):
                raise RuntimeError(
                    f"数据集 nc={unwrap_model(trainer.model).model[-1].nc} 与 CA checkpoint "
                    f"nc={load_summary['nc']} 不一致"
                )
            if refine_only:
                trainer_digest = shared_state_digest(unwrap_model(trainer.model).state_dict())
                if trainer_digest != load_summary["shared_digest"]:
                    raise RuntimeError(
                        "trainer setup 后的共享参数摘要与纯 CA 初始化不一致，拒绝开始 refine_only 训练"
                    )
            sync_obbrefine_runtime_attrs(
                trainer.model,
                trainer.args.aux_geo_ar,
                trainer.args.aux_geo_ws,
                True,
                trainer.args.refine_feature_detach,
                trainer.args.refine_experiment,
                trainer.args.refine_delta_max,
                trainer.args.refine_target_limit,
                trainer.args.refine_smooth_l1_beta,
                trainer.args.refine_identity_gain,
            )
            if hasattr(trainer, "ema") and trainer.ema is not None:
                sync_obbrefine_runtime_attrs(
                    trainer.ema.ema,
                    trainer.args.aux_geo_ar,
                    trainer.args.aux_geo_ws,
                    True,
                    trainer.args.refine_feature_detach,
                    trainer.args.refine_experiment,
                    trainer.args.refine_delta_max,
                    trainer.args.refine_target_limit,
                    trainer.args.refine_smooth_l1_beta,
                    trainer.args.refine_identity_gain,
                )
            freeze_summary = configure_refine_only_training(trainer) if refine_only else None
            print(
                f"[*] OBBRefine 运行时属性已同步: AR>{float(trainer.args.aux_geo_ar)}, "
                f"short<{float(trainer.args.aux_geo_ws)}px, "
                f"refine_feature_detach={bool(trainer.args.refine_feature_detach)}, "
                f"experiment={trainer.args.refine_experiment}, "
                f"默认验证/推理=coarse-only"
            )
            if freeze_summary:
                print(
                    f"[*] refine_only 冻结完成: trainable_tensors={freeze_summary['trainable_tensors']}, "
                    f"trainable_parameters={freeze_summary['trainable_parameters']:,}, "
                    f"frozen_parameters={freeze_summary['frozen_parameters']:,}"
                )

        model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
        if refine_only:
            model.add_callback("on_train_batch_start", keep_frozen_batch_norm_eval)

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
        if not train_config["refine_feature_detach"]:
            raise ValueError("最终保留路线要求 refine_feature_detach=True。")
        refine_mode_label = "fully-decoupled"

        print("=" * 60)
        print(f"  模型 YAML/权重: {model_path or train_config['model']}")
        print(f"  初始化: {'纯 CA checkpoint + 仅训练 cv5' if refine_only else '从 YAML 随机初始化'}")
        if refine_only:
            print(f"  CA 权重: {base_weights}")
        print(f"  数据集: {train_config['data']}")
        print(f"  训练轮数: {train_config['epochs']}")
        print(f"  批次大小: {train_config['batch']}")
        print(f"  图片尺寸: {train_config['imgsz']}")
        print(f"  输出目录: {train_config['project']}/{train_config['name']}")
        print(f"  临时日志文件: {log_file}")
        print(f"  aux_geo 增益: {train_config['aux_geo']}")
        print(f"  Refine Head: {train_config['refine_experiment']} (ne_refine=2, {refine_mode_label})")
        print(f"  refine_feature_detach: {train_config['refine_feature_detach']}")
        print("  验证/推理默认口径: coarse-only（默认禁用 refine inference）")
        print("  A/B 对照入口: python train_yolo11_obb_refine.py --mode val_ab --weights <ckpt>")
        print("=" * 60)

        # 5. 开始训练
        results = model.train(**train_config)
        save_dir = getattr(model.trainer, "save_dir", Path(train_config["project"]) / train_config["name"])

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
    elif cli_args.mode == "refine_only":
        experiment_config = get_refine_experiment_config(cli_args.refine_experiment)
        run_name = cli_args.name or (
            f"yolo11_obb_640_811_ca_refine_{experiment_config['run_version']}_"
            f"{cli_args.refine_experiment}_seed{cli_args.seed}"
        )
        main(
            train_overrides={
                "model": MODEL_V2_YAML,
                "data": cli_args.data,
                "epochs": 30,
                "batch": cli_args.batch,
                "imgsz": cli_args.imgsz,
                "device": cli_args.device,
                "workers": cli_args.workers,
                "project": cli_args.project,
                "name": run_name,
                "save_period": 5,
                "optimizer": "AdamW",
                "lr0": 3e-4,
                "warmup_epochs": 3.0,
                "seed": cli_args.seed,
                "pretrained": False,
                "refine_experiment": cli_args.refine_experiment,
                "refine_delta_max": experiment_config["refine_delta_max"],
                "refine_target_limit": experiment_config["refine_target_limit"],
                "refine_smooth_l1_beta": 0.02,
                "refine_identity_gain": 0.05,
                "aux_geo": 0.2,
                "refine_feature_detach": True,
            },
            refine_only=True,
            base_weights=cli_args.base_weights,
        )
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
