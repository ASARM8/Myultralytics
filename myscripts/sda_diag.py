"""
SDA-Head v2 推理诊断脚本

诊断 1：有效注入比 ρ = |αz| / |P|
  - 通过 forward hook 捕获中间特征 z4, z5, p4, p5
  - 计算 ρ = |α·z| / |P|，分 P4/P5、reg/angle 四组统计
  - 判断门控注入信号是否足够强以产生实际效果

诊断 3：强行开门实验
  - 手动将 gate 设为不同 alpha 值（0, 0.1, 0.2, 0.5, 1.0）
  - 跑完整验证（model.val()）获取 mAP 指标
  - 如果强行开门后变好 → 结构有价值，gate 学不会
  - 如果强行开门后更差 → 问题在分支语义质量

使用方法:
    修改下方 CONFIG 后运行: python myscripts/sda_diag.py
"""

import os
import sys
import glob
import yaml
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO

# ========================== 配置 ==========================
CONFIG = {
    # ---------- 模型 ----------
    "model": "/root/autodl-tmp/work-dirs/yolo11_obb-sda-v2/weights/best.pt",
    "model_label": "SDA v2",

    # ---------- 数据集 ----------
    "data": "/root/autodl-tmp/dataset/TTPLA-1024/dataset.yaml",

    # ---------- 推理参数 ----------
    "imgsz": 1024,
    "device": 0,
    "batch": 16,

    # ---------- 诊断 1：注入比 ----------
    "diag1_max_images": 100,  # 采样图片数（0=全部验证集）

    # ---------- 诊断 3：强行开门 ----------
    # 统一设置所有 gate 的 alpha 值列表
    "forced_alphas": [0.0, 0.1, 0.2, 0.5, 1.0],

    # ---------- 输出 ----------
    "output_dir": "/root/autodl-tmp/work-dirs/sda_diag_output",
}


# ========================== 工具函数 ==========================

def get_sda_head(model):
    """获取 AsymmetricOBB 检测头，验证是否为 SDA 模型。"""
    head = model.model.model[-1]
    if not hasattr(head, 'alpha_box_p4'):
        raise ValueError(
            "模型检测头不是 AsymmetricOBB，无法执行 SDA 诊断。"
            "请确认模型是使用 yolo11-obb-sda.yaml 训练的。"
        )
    return head


def get_gate_values(head):
    """读取当前 gate 参数值。"""
    return {
        "alpha_box_p4": head.alpha_box_p4.data.item(),
        "alpha_box_p5": head.alpha_box_p5.data.item(),
        "alpha_ang_p4": head.alpha_ang_p4.data.item(),
        "alpha_ang_p5": head.alpha_ang_p5.data.item(),
    }


def load_val_image_paths(data_yaml, max_images=0):
    """从 dataset.yaml 加载验证集图片路径。"""
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data_root = data.get("path", os.path.dirname(data_yaml))
    val_path = data.get("val", "images/val")
    if not os.path.isabs(val_path):
        val_path = os.path.join(data_root, val_path)
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    image_files = []
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(val_path, ext)))
    image_files.sort()
    if max_images > 0:
        image_files = image_files[:max_images]
    return image_files


def run_val(model, config):
    """运行完整验证，返回关键指标。"""
    metrics = model.val(
        data=config["data"],
        imgsz=config["imgsz"],
        device=config["device"],
        batch=config.get("batch", 16),
        verbose=False,
    )
    try:
        return {
            "mAP50": float(metrics.box.map50),
            "mAP75": float(metrics.box.map75),
            "mAP50-95": float(metrics.box.map),
        }
    except Exception:
        rd = metrics.results_dict
        return {
            "mAP50": float(rd.get("metrics/mAP50(B)", 0)),
            "mAP75": 0.0,
            "mAP50-95": float(rd.get("metrics/mAP50-95(B)", 0)),
        }


# ========================== 诊断 1：有效注入比 ==========================

def diag1_injection_ratio(model, config):
    """诊断 1：有效注入比 ρ = |αz| / |P|

    通过 hook 捕获 AsymmetricOBB.forward_head 中的中间特征：
      - p4, p5: 来自 FPN 的原始多尺度特征（head 输入）
      - z4, z5: fuse_p4/p5 的输出（融合后的辅助特征）
    然后计算 ρ = 通道平均 |α·z(h,w)| / |P(h,w)| 在所有空间位置上的统计量。
    """
    print("\n" + "=" * 60)
    print("  诊断 1：有效注入比 ρ = |αz| / |P|")
    print("=" * 60)

    head = get_sda_head(model)
    gates = get_gate_values(head)
    print(f"  当前 gate 值:")
    for k, v in gates.items():
        print(f"    {k} = {v:.6f}")

    # --- 注册 hooks ---
    captured = {}
    hooks = []

    def head_pre_hook(module, args):
        """捕获 head 输入：多尺度特征列表 [P3, P4, P5]。"""
        x = args[0]
        captured['p4'] = x[1].detach()
        captured['p5'] = x[2].detach()

    def fuse_output_hook(key):
        """捕获 fuse 层输出（z4 或 z5）。"""
        def hook_fn(module, inp, output):
            captured[key] = output.detach()
        return hook_fn

    hooks.append(head.register_forward_pre_hook(head_pre_hook))
    hooks.append(head.fuse_p4.register_forward_hook(fuse_output_hook('z4')))
    hooks.append(head.fuse_p5.register_forward_hook(fuse_output_hook('z5')))

    # --- 逐图推理，收集 ρ ---
    image_files = load_val_image_paths(config["data"], config.get("diag1_max_images", 100))
    print(f"  采样图片数: {len(image_files)}")

    # 四组统计：box_p4, box_p5, ang_p4, ang_p5
    # 每张图得到一组 ρ 的空间统计
    rho_per_image = defaultdict(list)     # key → [per-image mean ρ]
    rho_spatial_all = defaultdict(list)   # key → [所有空间位置的 ρ 值]（用于 median/max）
    signal_per_image = defaultdict(list)  # key → [per-image mean |αz|]
    feat_per_image = defaultdict(list)    # key → [per-image mean |P|]

    eps = 1e-8

    for idx, img_path in enumerate(image_files):
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(image_files)}] {os.path.basename(img_path)}")

        captured.clear()
        with torch.no_grad():
            model.predict(img_path, imgsz=config["imgsz"],
                         device=config["device"], verbose=False)

        if 'p4' not in captured or 'z4' not in captured:
            continue

        p4, p5 = captured['p4'], captured['p5']
        z4, z5 = captured['z4'], captured['z5']

        for level, p, z in [("p4", p4, z4), ("p5", p5, z5)]:
            # 逐空间位置计算：通道维度取平均绝对值
            p_strength = p.abs().mean(dim=1)      # (B, H, W)
            z_strength = z.abs().mean(dim=1)      # (B, H, W)

            feat_per_image[level].append(p_strength.mean().item())

            for branch in ["box", "ang"]:
                alpha_key = f"alpha_{branch}_{level}"
                alpha = abs(gates[alpha_key])
                az_map = alpha * z_strength           # (B, H, W) |α·z|
                rho_map = az_map / (p_strength + eps)  # (B, H, W)

                rho_per_image[f"{branch}_{level}"].append(rho_map.mean().item())
                signal_per_image[f"az_{branch}_{level}"].append(az_map.mean().item())

                # 采样空间位置用于 median/max（避免内存爆炸，每图最多采样 1000 点）
                flat = rho_map.reshape(-1)
                if flat.numel() > 1000:
                    indices = torch.randperm(flat.numel())[:1000]
                    flat = flat[indices]
                rho_spatial_all[f"{branch}_{level}"].extend(flat.cpu().tolist())

    # --- 移除 hooks ---
    for h in hooks:
        h.remove()

    # --- 汇总报告 ---
    print(f"\n  {'组别':>12s}  {'α 值':>10s}  {'ρ mean':>10s}  {'ρ median':>10s}  "
          f"{'ρ max':>10s}  {'|αz| mean':>10s}  {'|P| mean':>10s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    results = {}
    for key in ["box_p4", "box_p5", "ang_p4", "ang_p5"]:
        if not rho_per_image[key]:
            continue

        level = key.split("_")[1]
        branch = key.split("_")[0]
        alpha_key = f"alpha_{branch}_{level}"
        alpha_val = gates[alpha_key]

        rho_means = np.array(rho_per_image[key])
        spatial_vals = np.array(rho_spatial_all[key])
        az_means = np.array(signal_per_image[f"az_{key}"])
        p_means = np.array(feat_per_image[level])

        results[key] = {
            "alpha": alpha_val,
            "rho_mean": float(rho_means.mean()),
            "rho_median": float(np.median(spatial_vals)) if len(spatial_vals) else 0,
            "rho_max": float(spatial_vals.max()) if len(spatial_vals) else 0,
            "rho_std": float(rho_means.std()),
            "az_mean": float(az_means.mean()),
            "p_mean": float(p_means.mean()),
        }
        r = results[key]
        print(f"  {key:>12s}  {r['alpha']:10.6f}  {r['rho_mean']:10.6f}  "
              f"{r['rho_median']:10.6f}  {r['rho_max']:10.6f}  "
              f"{r['az_mean']:10.6f}  {r['p_mean']:10.4f}")

    # --- 诊断结论 ---
    print(f"\n  [诊断结论]")
    for key in ["box_p4", "box_p5", "ang_p4", "ang_p5"]:
        if key not in results:
            continue
        rho = results[key]["rho_mean"]
        alpha = results[key]["alpha"]
        if abs(alpha) < 1e-5:
            verdict = "⚠️  gate 仍为 ~0，注入完全无效（可能未学到有用信号）"
        elif rho < 0.001:
            verdict = "⚠️  注入几乎为零（α 非零但 z 本身很小）"
        elif rho < 0.01:
            verdict = "注入信号极弱，可能不足以影响预测"
        elif rho < 0.1:
            verdict = "注入信号较弱但可测量"
        elif rho < 0.5:
            verdict = "✓ 注入信号适中"
        else:
            verdict = "注入信号较强（注意是否过强导致不稳定）"
        print(f"    {key}: α={alpha:.6f}, ρ={rho:.6f} → {verdict}")

    return results


# ========================== 诊断 3：强行开门实验 ==========================

def diag3_forced_gate(model, config):
    """诊断 3：手动设置 alpha 值后重新验证。

    关键判定逻辑：
      - 强行开门后变好 → 结构有价值，gate 学不会（需调整 gate 训练策略）
      - 强行开门后更差 → 问题在分支语义质量，结构本身需要反思
      - 变化很小 → 当前数据集/场景下 SDA 结构收益有限
    """
    print("\n" + "=" * 60)
    print("  诊断 3：强行开门实验")
    print("=" * 60)

    head = get_sda_head(model)

    # 保存原始 gate 值（用于恢复）
    original_gates = {}
    for name in ["alpha_box_p4", "alpha_box_p5", "alpha_ang_p4", "alpha_ang_p5"]:
        original_gates[name] = getattr(head, name).data.clone()

    orig_vals = get_gate_values(head)
    print(f"  原始 gate 值:")
    for k, v in orig_vals.items():
        print(f"    {k} = {v:.6f}")

    forced_alphas = config.get("forced_alphas", [0.0, 0.1, 0.2, 0.5, 1.0])

    all_results = []

    # --- 实验 0：原始 gate 值 ---
    print(f"\n  [0] 使用原始 gate 值 (learned)...")
    m = run_val(model, config)
    all_results.append({"label": "原始 (learned)", "config": "learned", **orig_vals, **m})
    print(f"      mAP50={m['mAP50']:.4f}  mAP75={m['mAP75']:.4f}  mAP50-95={m['mAP50-95']:.4f}")

    # --- 实验 1~N：统一设置所有 gate ---
    for i, alpha_val in enumerate(forced_alphas):
        print(f"\n  [{i+1}] alpha = {alpha_val}（所有 gate 统一）...")
        with torch.no_grad():
            head.alpha_box_p4.fill_(alpha_val)
            head.alpha_box_p5.fill_(alpha_val)
            head.alpha_ang_p4.fill_(alpha_val)
            head.alpha_ang_p5.fill_(alpha_val)

        m = run_val(model, config)
        all_results.append({
            "label": f"alpha={alpha_val}",
            "config": f"all={alpha_val}",
            "alpha_box_p4": alpha_val, "alpha_box_p5": alpha_val,
            "alpha_ang_p4": alpha_val, "alpha_ang_p5": alpha_val,
            **m,
        })
        print(f"      mAP50={m['mAP50']:.4f}  mAP75={m['mAP75']:.4f}  mAP50-95={m['mAP50-95']:.4f}")

    # --- 实验 N+1：只开 box gate ---
    print(f"\n  [{len(forced_alphas)+1}] 只开 box gate (box=0.2, ang=0)...")
    with torch.no_grad():
        head.alpha_box_p4.fill_(0.2)
        head.alpha_box_p5.fill_(0.2)
        head.alpha_ang_p4.fill_(0.0)
        head.alpha_ang_p5.fill_(0.0)
    m = run_val(model, config)
    all_results.append({
        "label": "只开 box (0.2)",
        "config": "box=0.2,ang=0",
        "alpha_box_p4": 0.2, "alpha_box_p5": 0.2,
        "alpha_ang_p4": 0.0, "alpha_ang_p5": 0.0,
        **m,
    })
    print(f"      mAP50={m['mAP50']:.4f}  mAP75={m['mAP75']:.4f}  mAP50-95={m['mAP50-95']:.4f}")

    # --- 实验 N+2：只开 angle gate ---
    print(f"\n  [{len(forced_alphas)+2}] 只开 angle gate (box=0, ang=0.2)...")
    with torch.no_grad():
        head.alpha_box_p4.fill_(0.0)
        head.alpha_box_p5.fill_(0.0)
        head.alpha_ang_p4.fill_(0.2)
        head.alpha_ang_p5.fill_(0.2)
    m = run_val(model, config)
    all_results.append({
        "label": "只开 ang (0.2)",
        "config": "box=0,ang=0.2",
        "alpha_box_p4": 0.0, "alpha_box_p5": 0.0,
        "alpha_ang_p4": 0.2, "alpha_ang_p5": 0.2,
        **m,
    })
    print(f"      mAP50={m['mAP50']:.4f}  mAP75={m['mAP75']:.4f}  mAP50-95={m['mAP50-95']:.4f}")

    # --- 恢复原始 gate 值 ---
    with torch.no_grad():
        for name, val in original_gates.items():
            getattr(head, name).data.copy_(val)
    print(f"\n  [*] 已恢复原始 gate 值")

    # --- 汇总表格 ---
    baseline = all_results[0]
    print(f"\n  {'配置':>20s}  {'mAP50':>8s}  {'mAP75':>8s}  {'mAP50-95':>10s}  "
          f"{'Δ50':>8s}  {'Δ75':>8s}  {'Δmap':>8s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in all_results:
        d50 = r["mAP50"] - baseline["mAP50"]
        d75 = r["mAP75"] - baseline["mAP75"]
        dmap = r["mAP50-95"] - baseline["mAP50-95"]
        print(f"  {r['label']:>20s}  {r['mAP50']:8.4f}  {r['mAP75']:8.4f}  "
              f"{r['mAP50-95']:10.4f}  {d50:+8.4f}  {d75:+8.4f}  {dmap:+8.4f}")

    # --- 诊断结论 ---
    print(f"\n  [诊断结论]")
    # 排除 alpha=0 的实验（等价于关闭注入）
    non_zero_results = [r for r in all_results[1:] if r.get("config", "") != "all=0.0"]
    if non_zero_results:
        best = max(non_zero_results, key=lambda r: r["mAP50-95"])
        worst = min(non_zero_results, key=lambda r: r["mAP50-95"])
        delta_best = best["mAP50-95"] - baseline["mAP50-95"]
        delta_worst = worst["mAP50-95"] - baseline["mAP50-95"]

        if delta_best > 0.002:
            print(f"    ✓ 强行开门后指标提升！最佳配置: {best['label']}")
            print(f"      mAP50-95: {baseline['mAP50-95']:.4f} → {best['mAP50-95']:.4f} "
                  f"({delta_best:+.4f})")
            print(f"    → 结构有价值，但 gate 学习策略需调整（考虑更大初始值/更高 LR/warmup）")
        elif delta_worst < -0.005:
            print(f"    ⚠️  开门后指标显著下降！最差配置: {worst['label']}")
            print(f"      mAP50-95: {baseline['mAP50-95']:.4f} → {worst['mAP50-95']:.4f} "
                  f"({delta_worst:+.4f})")
            print(f"    → 分支语义质量有问题，融合特征可能引入了噪声")
        else:
            print(f"    注入对指标影响很小（最大变化 {delta_best:+.4f}）")
            print(f"    → 当前数据集/场景下 SDA 结构收益有限，或融合分支未学到有用信息")

    # alpha=0 vs 原始的对比（检查 gate 是否在帮倒忙）
    alpha0_result = next((r for r in all_results if r.get("config") == "all=0.0"), None)
    if alpha0_result:
        d0 = alpha0_result["mAP50-95"] - baseline["mAP50-95"]
        if d0 > 0.002:
            print(f"    ⚠️  关闭注入 (α=0) 后指标反而更好 ({d0:+.4f}) → 当前注入在帮倒忙！")
        elif d0 < -0.002:
            print(f"    ✓ 关闭注入后指标下降 ({d0:+.4f}) → 学到的注入确实有帮助")
        else:
            print(f"    关闭注入对指标几乎无影响 ({d0:+.4f}) → gate 学到的值太小，注入未生效")

    return all_results


# ========================== 报告生成 ==========================

def generate_report(config, diag1_result, diag3_result, output_dir):
    """生成诊断 Markdown 报告。"""
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    lines.append("# SDA-Head v2 推理诊断报告")
    lines.append("")
    lines.append(f"**模型**: `{config['model']}`  ")
    lines.append(f"**标签**: {config['model_label']}  ")
    lines.append(f"**数据集**: `{config['data']}`  ")
    lines.append("")

    # ---- 诊断 1 ----
    lines.append("## 诊断 1：有效注入比 ρ = |αz| / |P|")
    lines.append("")
    lines.append("ρ 衡量门控注入信号相对于原始特征的强度。ρ < 0.001 表示注入几乎无效。")
    lines.append("")
    lines.append("| 组别 | α 值 | ρ mean | ρ median | ρ max | \\|αz\\| mean | \\|P\\| mean |")
    lines.append("|:----:|:----:|:------:|:-------:|:-----:|:---------:|:-------:|")
    if diag1_result:
        for key in ["box_p4", "box_p5", "ang_p4", "ang_p5"]:
            if key not in diag1_result:
                continue
            r = diag1_result[key]
            lines.append(
                f"| {key} | {r['alpha']:.6f} | {r['rho_mean']:.6f} | "
                f"{r['rho_median']:.6f} | {r['rho_max']:.6f} | "
                f"{r['az_mean']:.6f} | {r['p_mean']:.4f} |"
            )
    lines.append("")

    # ---- 诊断 3 ----
    lines.append("## 诊断 3：强行开门实验")
    lines.append("")
    lines.append("手动设置 gate alpha 值后重新跑完整验证。")
    lines.append("")
    lines.append("| 配置 | mAP50 | mAP75 | mAP50-95 | Δ mAP50 | Δ mAP75 | Δ mAP50-95 |")
    lines.append("|:----:|:-----:|:-----:|:--------:|:-------:|:-------:|:----------:|")
    if diag3_result:
        baseline = diag3_result[0]
        for r in diag3_result:
            d50 = r["mAP50"] - baseline["mAP50"]
            d75 = r["mAP75"] - baseline["mAP75"]
            dmap = r["mAP50-95"] - baseline["mAP50-95"]
            lines.append(
                f"| {r['label']} | {r['mAP50']:.4f} | {r['mAP75']:.4f} | "
                f"{r['mAP50-95']:.4f} | {d50:+.4f} | {d75:+.4f} | {dmap:+.4f} |"
            )
    lines.append("")
    lines.append("**关键判定**：")
    lines.append("- 强行开门后变好 → 结构有价值，gate 学习策略需调整")
    lines.append("- 强行开门后更差 → 问题在分支语义质量")
    lines.append("- 关闭注入(α=0)后更好 → 当前注入在帮倒忙")
    lines.append("- 关闭注入后更差 → 学到的注入确实有帮助")
    lines.append("")

    report_path = os.path.join(output_dir, "sda_diag_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  报告已保存: {report_path}")
    return report_path


# ========================== 主函数 ==========================

def main():
    print("=" * 60)
    print("  SDA-Head v2 推理诊断")
    print("=" * 60)
    print(f"\n  加载模型: {CONFIG['model']}")

    model = YOLO(CONFIG["model"])

    head = get_sda_head(model)
    print(f"  检测头类型: {type(head).__name__}")

    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 诊断 1：有效注入比
    diag1_result = diag1_injection_ratio(model, CONFIG)

    # 诊断 3：强行开门实验
    diag3_result = diag3_forced_gate(model, CONFIG)

    # 生成报告
    generate_report(CONFIG, diag1_result, diag3_result, output_dir)

    print("\n" + "=" * 60)
    print("  诊断完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
