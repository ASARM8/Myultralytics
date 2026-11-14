"""Build midterm tables and scientific figures from downloaded experiment outputs.

Uses existing NumPy, Pillow and Matplotlib only. It never loads model weights,
trains a model, changes the Word manuscript or rewrites source experiment data.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon, Patch
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "mydocs/中期报告/图表补充_20261114"
DET = ROOT / "res/640-811/ivc_low_gain_evidence_20260817_224706"
SEG = ROOT / "res/innovation2"
FAIL = SEG / "failure_diagnostics_val_20261112_150721"
EFF = SEG / "efficiency_axes_val_20261114_070523"
COLORS = {"target": "#697386", "points": "#2874A6", "boxes": "#D18B24", "axes": "#B43E58"}
ORDER = list(COLORS)
SOURCES, FIGURES, TABLES = {}, [], []


def source(path):
    path = Path(path)
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(path):
    return json.loads(source(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with source(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(name, rows):
    path = OUT / "tables" / f"{name}.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def cell(value):
    return f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value)


def add_table(name, title, rows, note):
    write_csv(name, rows)
    TABLES.append((name, title, rows, note))


def save_fig(fig, name, title, caption):
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT / "figures" / f"{name}.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIGURES.append((name, title, caption))


def style():
    for path in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "pdf.fonttype": 42, "svg.fonttype": "none"})


def detection_table():
    raw = read_csv(DET / "main_results_low_gain.csv")
    rows = [{"方法": r["method"], **{label: float(r[key])*100 for label, key in
             (("Precision (%)", "precision"), ("Recall (%)", "recall"), ("OBB mAP50 (%)", "map50"),
              ("OBB mAP50–95 (%)", "map50_95"))}} for r in raw]
    note = "同一验证集的阶段性对照；CA指回归区间扩展与范围感知分配的联合配置。Refine倍率α=0.42。该表不能单独分离reg_max与分配机制的贡献。"
    add_table("02_detection_results", "表2 创新点一检测结果", rows, note)
    fig, ax = plt.subplots(figsize=(10, 2.3))
    ax.axis("off")
    table = ax.table(cellText=[[cell(v) for v in r.values()] for r in rows],
                     colLabels=list(rows[0]), loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for (r, c), item in table.get_celld().items():
        item.set_edgecolor("#D0D5DC")
        if r == 0:
            item.set_facecolor("#EAF0F5")
            item.set_text_props(weight="bold")
    save_fig(fig, "02_detection_results", "创新点一检测结果表", note)


def threshold_figures():
    scans = {}
    for path in list(SEG.glob("threshold_scan*.json")) + [SEG / "axes/threshold_scan_val.json"]:
        payload = read_json(path)
        scans[payload["prompt"]] = payload
    assert set(scans) == set(ORDER)
    assert len({p["cache_sha256"] for p in scans.values()}) == 1
    records, table = [], []
    for prompt in ORDER:
        for r in scans[prompt]["results"]:
            exposed = r["matched_with_neighbors"]
            records.append({"prompt": prompt, "threshold": r["mask_threshold"],
                "mask_map50_95_pct": 100*r["mask_mAP50_95"],
                "neighbor_subset_recall_pct": 100*exposed["target_pixel_recall"],
                "neighbor_subset_leakage_pct": 100*exposed["neighbor_pixel_leakage"],
                "matched_mean_iou_pct": 100*r["matched"]["mean_instance_iou"],
                "pair_success_iou50": r["neighbor_pairs"]["all"]["by_mask_iou"]["0.50"][prompt]["successful_pairs"],
                "pair_success_iou75": r["neighbor_pairs"]["all"]["by_mask_iou"]["0.75"][prompt]["successful_pairs"]})
        r = next(x for x in records if x["prompt"] == prompt and x["threshold"] == .5)
        table.append({"方法": prompt, "Mask mAP50–95 (%)": r["mask_map50_95_pct"],
                      "邻线子集目标召回 (%)": r["neighbor_subset_recall_pct"],
                      "邻线子集泄漏 (%)": r["neighbor_subset_leakage_pct"],
                      "双实例IoU≥0.5的邻线对": r["pair_success_iou50"]})
    write_csv("07_threshold_scan_all", records)
    add_table("03_prompt_results_without_swap", "表3 四种提示的已有定量结果", table,
              "掩码阈值0.5；mAP为全验证集，召回与泄漏为386个暴露于邻线的匹配实例，邻线对总数270。互换结果需归档四组metrics_val.json，不能用这里的邻线对数替代。")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for prompt in ORDER:
        rows = [r for r in records if r["prompt"] == prompt]
        x = [r["neighbor_subset_recall_pct"] for r in rows]
        y = [r["neighbor_subset_leakage_pct"] for r in rows]
        axes[0].plot(x, y, "o-", label=prompt, color=COLORS[prompt], lw=1.7, ms=5)
        axes[1].plot([r["threshold"] for r in rows], [r["mask_map50_95_pct"] for r in rows],
                     "o-", label=prompt, color=COLORS[prompt], lw=1.7, ms=5)
        if prompt == "axes":
            for a, b, r in zip(x, y, rows):
                axes[0].annotate(f"t={r['threshold']:.1f}", (a, b), xytext=(5, -13),
                                 textcoords="offset points", fontsize=8, color=COLORS[prompt])
    axes[0].set(xlabel="邻线子集目标像素召回率 (%)", ylabel="邻线像素泄漏率 (%)",
                title="(a) 目标保留与邻线泄漏的工作点", xlim=(70, 94), ylim=(2.1, 5.5))
    axes[1].set(xlabel="掩码二值化阈值 t", ylabel="Mask mAP50–95 (%)", title="(b) 完整验证集的阈值敏感性",
                xticks=[.1, .2, .3, .4, .5])
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    save_fig(fig, "07a_recall_leakage_threshold", "目标召回、邻线泄漏与阈值敏感性",
             "左图统计386个邻线暴露实例；右图统计完整1695图块。点对应实际阈值0.1至0.5，连线仅辅助阅读，不代表未测阈值。axes相对target与points的泄漏减少需结合召回下降解释；不能据此断言全面优于boxes。")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), layout="constrained")
    for ax, metric, title in zip(axes, ("pair_success_iou50", "pair_success_iou75"), ("0.50", "0.75")):
        for prompt in ORDER:
            rows = [r for r in records if r["prompt"] == prompt]
            ax.plot([r["threshold"] for r in rows], [r[metric] for r in rows], "o-",
                    color=COLORS[prompt], label=prompt)
        ax.set(xlabel="掩码二值化阈值 t", ylabel="成功邻线对数 / 270", title=f"两根电线的掩码IoU均≥{title}",
               xticks=[.1, .2, .3, .4, .5])
        ax.legend(frameon=False, ncol=2)
        ax.grid(alpha=.2)
    save_fig(fig, "07b_neighbor_pair_threshold", "常规推理中邻线对成功数随阈值变化",
             "分母为270个无序真值邻线对，包含81对至少一根未匹配到候选的情况。这是常规实例分割的绝对质量评价，不是固定裁剪的角色互换试验。")
    return records


def paired_figures():
    data = read_json(SEG / "axes/paired_geometry_val.json")
    stats = data["matched"]
    rows = [{"指标": "Mask mAP50–95 (%)", "粗框": data["mask_mAP50_95"]["coarse"]*100,
             "精修框": data["mask_mAP50_95"]["refined"]*100,
             "变化（百分点）": data["mask_mAP50_95_delta"]*100}]
    for key, label in (("mean_tile_obb_iou", "平均框IoU (%)"), ("mean_instance_iou", "平均掩码IoU (%)"),
                       ("target_crop_coverage", "目标裁剪覆盖率 (%)"), ("target_pixel_recall", "目标像素召回 (%)"),
                       ("neighbor_pixel_leakage", "邻线泄漏 (%)"),
                       ("swap_both_correct_fraction", "固定粗框邻线队列互换正确率 (%)")):
        rows.append({"指标": label, "粗框": stats["coarse"][key]*100, "精修框": stats["refined"][key]*100,
                     "变化（百分点）": stats["refined_minus_coarse"][key]*100})
    add_table("08_paired_geometry", "表8 粗框与精修框的配对串联结果", rows,
              "固定axes权重；匹配实例3140，互换试验346次。分割器使用精修框训练，结果包含训练输入适配影响。这里的346次不能与四提示对照的342次直接合并。")
    records = [{"tile": r["tile"], "proposal_id": r["proposal_id"], "truth_id": r["truth_id"],
                "exposed": r["exposed"], "box_iou_delta_pp": 100*(r["refined"]["tile_obb_iou"]-r["coarse"]["tile_obb_iou"]),
                "mask_iou_delta_pp": 100*r["mask_iou_delta"]} for r in data["paired_instances"]]
    write_csv("08_paired_instance_deltas", records)
    x, y = np.array([[r["box_iou_delta_pp"], r["mask_iou_delta_pp"]] for r in records]).T
    exposed = np.array([r["exposed"] for r in records])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained", gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.scatter(x[~exposed], y[~exposed], s=9, alpha=.3, color="#718096", label="其余匹配实例")
    ax.scatter(x[exposed], y[exposed], s=15, alpha=.55, color=COLORS["axes"], label="固定粗框邻线子集")
    ax.axhline(0, color="#333333", lw=.8)
    ax.axvline(0, color="#333333", lw=.8)
    ax.set(xlabel="精修后框IoU变化（百分点）", ylabel="精修后掩码IoU变化（百分点）",
           title="(a) 同一候选的几何变化与分割变化")
    ax.legend(frameon=False, fontsize=9)
    names = ["improved", "unchanged", "worsened"]
    matrix = np.array([[stats["box_iou_change_by_mask_iou_change"][a][b] for b in names] for a in names])
    axes[1].imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=13,
                         color="white" if matrix[i, j] > matrix.max()*.55 else "#152A3C")
    labels = ["提高", "不变", "下降"]
    axes[1].set(xticks=range(3), xticklabels=labels, yticks=range(3), yticklabels=labels,
                xlabel="掩码IoU变化", ylabel="框IoU变化", title="(b) 配对变化计数（3140实例）")
    save_fig(fig, "08_paired_geometry_changes", "检测精修向分割传递的配对分析",
             "全部3140个匹配实例均保留。1818个实例掩码IoU提高，86个不变，1236个下降；框IoU提高并不必然使掩码IoU提高。散点与计数表示配对关联，不构成因果或独立泛化证明。")
    return data


def quad_txt(path, shape):
    rows = []
    for line in source(path).read_text().splitlines():
        values = line.split()
        if values:
            points = np.array(list(map(float, values[1:9]))).reshape(4, 2)
            points *= np.array([shape[1], shape[0]])
            rows.append(points)
    return rows


def detection_examples():
    manifest = read_csv(DET / "qualitative_figure/fig6_qualitative_manifest.csv")
    selected = [manifest[i] for i in (0, 1, 3)]
    fig, axes = plt.subplots(3, 4, figsize=(13, 5.7), layout="constrained")
    labels = ["真值", "Baseline + 真值", "CA + 真值", "CA+Refine + 真值"]
    for row, sample in enumerate(selected):
        image_path = DET / "qualitative_predictions/images" / Path(sample["image_path"]).name
        image = np.asarray(Image.open(source(image_path)).convert("RGB"))
        gt = quad_txt(DET / "qualitative_predictions/gt" / Path(sample["gt_path"]).name, image.shape)
        center = np.concatenate(gt).mean(axis=0)
        cx, cy = np.clip(center, [200, 100], [image.shape[1]-200, image.shape[0]-100])
        for col, folder in enumerate(("gt", "baseline", "ca", "refined")):
            ax = axes[row, col]
            ax.imshow(image)
            for points in gt:
                ax.add_patch(Polygon(points, closed=True, fill=False, edgecolor="#26C281", lw=1.1,
                                     linestyle="-" if col == 0 else "--"))
            if col:
                key = {"baseline": "baseline_path", "ca": "ca_path", "refined": "final_path"}[folder]
                for points in quad_txt(DET / "qualitative_predictions" / folder / Path(sample[key]).name, image.shape):
                    ax.add_patch(Polygon(points, closed=True, fill=False, edgecolor="#FFB000", lw=1.2))
            ax.set_xlim(cx-200, cx+200)
            ax.set_ylim(cy+100, cy-100)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(labels[col])
            if col == 0:
                ax.set_ylabel(f"案例{row+1}\nΔIoU={float(sample['matched_iou_gain']):+.3f}")
            if col == 1:
                ax.set_xlabel(Path(sample["original_image_path"]).stem, fontsize=7)
    save_fig(fig, "09a_detection_examples", "旋转框精修改善与退化案例",
             "绿色为真值旋转框，橙色为对应方法预测；每行使用相同原图局部窗口。采用已归档清单中的两个改善案例和一个退化案例，ΔIoU沿用原图匹配统计。局部窗口只用于显示，不重新计算整图指标。")


def failure_examples():
    records = []
    fig, axes = plt.subplots(3, 4, figsize=(12, 9.5), layout="constrained")
    # Explicitly show recovery, unresolved under-segmentation, and over-segmentation.
    for row, prefix in enumerate(("003_", "005_", "001_")):
        folder = next((FAIL / "cases").glob(prefix + "*"))
        meta = read_json(folder / "case.json")
        with np.load(source(folder / "arrays.npz")) as arrays:
            gt, valid = arrays["gt"], arrays["valid"]
            p5, p2 = arrays["pred"] & valid, (arrays["probability_tile"] >= .2) & valid
            assert np.array_equal(p5, (arrays["probability_tile"] >= .5) & valid)
        image = np.asarray(Image.open(source((folder / meta["original_image"]).resolve())).convert("RGB"))
        ys, xs = np.where(gt & ~p5)
        cx, cy = int(np.median(xs)), int(np.median(ys))
        left, top = np.clip([cx-128, cy-128], 0, [image.shape[1]-256, image.shape[0]-256])
        roi = (slice(top, top+256), slice(left, left+256))
        masks = [None, gt, p5, p2]
        for col, mask in enumerate(masks):
            panel = image[roi].astype(float) / 255
            if mask is not None:
                color = np.array([.1, .85, .45] if col == 1 else [1., .63, .05])
                keep = mask[roi]
                panel[keep] = .35*panel[keep]+.65*color
            axes[row, col].imshow(panel, interpolation="nearest")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(["原图局部", "目标真值", "axes预测 t=0.5", "axes预测 t=0.2"][col])
        iou2 = np.count_nonzero(p2 & gt) / max(1, np.count_nonzero(p2 | gt))
        axes[row, 0].set_ylabel(["阈值降低后恢复", "降低阈值仍漏分", "降低阈值扩大误分"][row])
        axes[row, 2].set_xlabel(f"整图实例IoU={meta['mask_iou']:.3f}")
        axes[row, 3].set_xlabel(f"整图实例IoU={iou2:.3f}")
        records.append({"case": meta["case"], "tile": meta["tile"], "iou_threshold_05": meta["mask_iou"],
                        "iou_threshold_02": iou2, "zoom_left": int(left), "zoom_top": int(top), "zoom_size": 256})
    write_csv("09_failure_example_manifest", records)
    summary = read_json(FAIL / "summary.json")
    cases = summary["cases"]
    inside = sum(r["missed_inside_crop"] for r in cases)
    outside = sum(r["missed_outside_crop"] for r in cases)
    save_fig(fig, "09b_segmentation_failure_examples", "分割阈值变化的恢复与失败案例",
             "同一冻结axes输出概率图分别以0.5和0.2二值化，无新增推理或训练。每行显示同一256×256局部窗口，标注的IoU按完整640×640实例计算。三例来自37个失败实例，仅用于解释不同误差类型，不代表全验证集平均收益。"
             f"在{len(cases)}个失败实例中，遗漏目标像素的{100*inside/(inside+outside):.2f}%（{inside:,}个）位于裁剪内部，"
             f"{100*outside/(inside+outside):.2f}%（{outside:,}个）位于裁剪外部。"
             "该子集的主要漏分发生在已保留的图像范围内，后续应优先检查预测响应与阈值选择；这一比例不能直接推广至整个验证集。")


def alpha_figure():
    records = []
    for path in sorted((ROOT / "res/640-811/refine_v311_low_gain").glob("*/val_metrics.csv")):
        raw = next(r for r in read_csv(path) if r["variant"] == "refined")
        records.append({"alpha": float(raw["residual_scale"]), "map50_pct": float(raw["map50"])*100,
                        "map50_95_pct": float(raw["map50_95"])*100, "ap95_pct": float(raw["ap95"])*100})
    records.sort(key=lambda r: r["alpha"])
    write_csv("10_residual_scale_sweep", records)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), layout="constrained")
    for ax, key, label in zip(axes, ("map50_95_pct", "ap95_pct"), ("OBB mAP50–95 (%)", "OBB AP95 (%)")):
        ax.plot([r["alpha"] for r in records], [r[key] for r in records], "o-", color="#2874A6", ms=5)
        current = next(r for r in records if r["alpha"] == .42)
        ax.axvline(.42, color=COLORS["axes"], linestyle="--", lw=1.2, label="当前串联配置 α=0.42")
        ax.scatter([.42], [current[key]], s=65, color=COLORS["axes"], zorder=5)
        ax.set(xlabel="尺度残差倍率 α", ylabel=label, xlim=(-.03, 1.03))
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_title("(a) 整体检测精度")
    axes[1].set_title("(b) 高IoU阈值下的定位精度")
    save_fig(fig, "10_residual_scale_sensitivity", "统一配置下的精修倍率敏感性",
             "用户已确认扫描配置统一。绘出全部12个倍率；α=0表示不应用残差，α=1表示完整应用有界残差。α=0.42是当前串联采用的配置，非本次扫描mAP最优点；α=1的mAP更高，AP95略低。不能以此图声称0.42全面最优或已证明更稳定。")
    return records


def efficiency_figures():
    data, manifest = read_json(EFF / "summary.json"), read_json(EFF / "manifest.json")
    raw = read_csv(EFF / "per_image.csv")
    end = [r for r in raw if r["phase"] == "end_to_end"]
    observed = np.array([float(r["total_ms"]) for r in end])
    assert len(end) == 5085 and abs(observed.mean()-data["end_to_end_ms"]["mean"]) < 1e-8
    stages = data["stage_ms_separate_pass"]
    names = ["图像读取", "检测输入准备", "检测与旋转NMS", "尺度精修", "几何回传", "裁剪与提示", "分割推理与回传", "掩码恢复"]
    values = [v["mean"] for v in stages.values()]
    total = data["stage_instrumented_total_ms"]["mean"]
    stage_rows = [{"阶段": name, "平均耗时(ms)": value, "占分阶段轮总时延(%)": value/total*100}
                  for name, value in zip(names, values)]
    add_table("efficiency_stages", "效率补充：各阶段耗时", stage_rows,
              "独立的分阶段同步计时轮次，1695图块。不能将各阶段P95相加；主端到端指标来自另外三轮自然执行。")
    fig, ax = plt.subplots(figsize=(8, 4.7), layout="constrained")
    ax.barh(names[::-1], values[::-1], color=["#A3B1C2"]*2+["#B43E58", "#D18B24"]+["#A3B1C2"]*4)
    for i, v in enumerate(values[::-1]):
        ax.text(v+.15, i, f"{v:.2f} ms ({v/total*100:.1f}%)", va="center", fontsize=9)
    ax.set(xlabel="单图块平均耗时 (ms)", xlim=(0, max(values)*1.4), title="完整流程的阶段耗时分解（独立计时轮）")
    save_fig(fig, "E01_stage_latency", "效率补充：阶段耗时分解",
             "RTX 4090、FP32张量、单图块、逐候选分割，检测输入640×640。分割推理项含输入传输和概率图回传。阶段百分比以分阶段轮总时延为分母，未归属少量调度开销保留在总时间中。")
    grouped = {}
    for r in end:
        grouped.setdefault(r["tile"], []).append(r)
    points = [{"tile": k, "proposals": int(v[0]["proposals"]), "crop_pixels": int(v[0]["crop_pixels"]),
               "mean_latency_ms": np.mean([float(x["total_ms"]) for x in v])} for k, v in grouped.items()]
    write_csv("efficiency_per_tile_three_repeat_mean", points)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")
    for ax, key, label, scale in zip(axes, ("proposals", "crop_pixels"), ("候选电线数", "裁剪总面积（千像素）"), (1, 1000)):
        ax.scatter([p[key]/scale for p in points], [p["mean_latency_ms"] for p in points],
                   s=14, alpha=.38, color="#2874A6")
        ax.set(xlabel=label, ylabel="三轮平均端到端时延 (ms)")
        ax.grid(alpha=.2)
    save_fig(fig, "E02_workload_latency", "效率补充：候选规模与推理时延",
             "每点是一张图块在三轮中的平均时延，共1695点，避免将重复测量画成独立样本。所有候选均保留。相关关系用于定位候选调度与裁剪瓶颈，不单独证明某个因素的因果贡献。")
    return data, manifest


def write_documents(thresholds, paired, alpha, efficiency, environment):
    text = ["# 中期报告图表补充与结果解读", "", "本材料对应2026年11月14日的图表整理。原Word文档未改动。第1项数据设置表暂不制作；第4—6和11项只提供详细文字说明，不绘图。", "",
            "## 效率结果", "",
            f"完整流程共测量1695张图块、三轮5085次。平均时延{efficiency['end_to_end_ms']['mean']:.2f} ms，中位数{efficiency['end_to_end_ms']['median']:.2f} ms，P95为{efficiency['end_to_end_ms']['p95']:.2f} ms，P99为{efficiency['end_to_end_ms']['p99']:.2f} ms。平均时延倒数为{efficiency['serial_tiles_per_second']:.2f}图块/秒，属于串行采样的换算速度，不是完整航拍帧或相机系统吞吐率。", "",
            "在线候选与原缓存的1695次核对全部一致；3390次重复输出比较和1695次分阶段输出比较均无差异。GPU峰值已分配显存0.253 GiB，预留显存0.318 GiB，口径为PyTorch内存，不包含全部驱动和设备资源。没有测量功耗。", "",
            "三轮均值为39.13、36.73、36.91 ms，极差相对均值为6.40%，高于预设5%门限。应保留三轮数据并注明稳定性未达门限，不能事后删除第一轮或放宽阈值来宣称通过。后二轮接近只能提示可能有启动/系统负载影响，不能确定原因。", "",
            "分阶段轮中，检测与旋转NMS约19.34 ms、分割推理与回传7.13 ms、裁剪与提示4.28 ms、精修2.08 ms。高候选数场景存在明显长尾：20根及以上候选共17张图块，组内平均时延230.47 ms。后续优化可优先分析逐候选调用、CPU提示构造与批次调度；这些结果尚未证明任何加速方案已有效。", "",
            "测试平台为RTX 4090，单CPU计算线程，输入640×640，三阶段模型保持加载，FP32张量；环境记录cuDNN允许TF32。不能直接与历史输入672×672、计时范围不同的检测专用结果作速度增减比较。", "",
            "## 已完成的表格", ""]
    for name, title, rows, note in TABLES:
        headers = list(rows[0])
        text += [f"### {title}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"]
        text += ["| " + " | ".join(cell(v) for v in r.values()) + " |" for r in rows]
        text += ["", note, "", f"原始精度表格：[CSV](tables/{name}.csv)", ""]
    text += ["## 图件与图注", ""]
    for name, title, caption in FIGURES:
        text += [f"### {title}", "", f"![{title}](figures/{name}.png)", "", caption, "",
                 f"[PNG](figures/{name}.png) · [SVG](figures/{name}.svg) · [PDF](figures/{name}.pdf)", ""]
    text += ["## 使用边界", "",
             "1. 第9项已生成检测改善/退化案例和axes阈值恢复/失败案例；简单的裁剪内外漏分统计并入失败案例图注，不单独绘图。本地没有四个分割模型在同一案例上的完整掩码图，也没有角色互换的两张输出掩码，暂未生成这两类可视化。四组原始互换报告的获取方式见另附说明。",
             "2. 第7项的整体mAP最佳采样阈值均为0.4，axes与points仅差约0.075个百分点。整体精度接近时，应主要讨论目标选择及召回—泄漏关系，不宣称所有指标均显著领先。",
             "3. 第8项固定分割器权重和候选身份，整体Mask mAP提升约1.254个百分点，但目标总体召回略降、固定粗框队列互换结果下降。所有方向均已保留。",
             "4. 第10项α=0.42对应49.581% OBB mAP50–95；α=1对应56.250%，AP95分别为9.932%和9.717%。0.42可表述为当前保守串联配置，不能表述为扫描最优倍率。更换倍率后的分割效果尚需验证。",
             "5. 本批图表没有改变数据划分或权重；来源一致性按用户确认处理。没有运行本地模型推理、训练或安装依赖。", ""]
    (OUT / "图表与结果说明.md").write_text("\n".join(text), encoding="utf-8")
    pages = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>中期报告图表补充</title>",
             "<style>body{font:16px/1.7 'Microsoft YaHei',sans-serif;max-width:1100px;margin:36px auto;padding:0 24px;color:#243044}img{max-width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd3dc;padding:9px;text-align:center}th{background:#edf2f6}section{margin:36px 0}h1,h2{color:#192e45}figcaption{margin:10px 0 24px}a{color:#256b9c}</style>",
             "<h1>中期报告图表补充</h1><p>检测对照、阈值分析、配对串联、案例、倍率扫描及效率结果。各PNG为300 dpi；SVG与PDF供缩放与排版。</p>"]
    for name, title, rows, note in TABLES:
        pages += [f"<section><h2>{html.escape(title)}</h2><table><tr>"+"".join(f"<th>{html.escape(k)}</th>" for k in rows[0])+"</tr>"]
        pages += ["<tr>"+"".join(f"<td>{html.escape(cell(v))}</td>" for v in r.values())+"</tr>" for r in rows]
        pages += [f"</table><p>{html.escape(note)}</p></section>"]
    for name, title, caption in FIGURES:
        pages += [f"<section><h2>{html.escape(title)}</h2><img src='figures/{name}.png'><figcaption>{html.escape(caption)}</figcaption></section>"]
    pages.append("</html>")
    (OUT / "图表预览.html").write_text("\n".join(pages), encoding="utf-8")
    (OUT / "数据来源与生成记录.json").write_text(json.dumps({
        "generator": str(Path(__file__).relative_to(ROOT)), "source_sha256": SOURCES,
        "figures": [r[0] for r in FIGURES], "tables": [r[0] for r in TABLES],
        "user_confirmed": {"same_weight_origin": True, "alpha_sweep_same_configuration": True},
        "not_generated": ["4 整体路线图", "5 创新点一方法图", "6 创新点二方法图", "11 部署图", "四方法同场景掩码及互换掩码图（缺原始输出）"]},
        ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    style()
    detection_table()
    thresholds = threshold_figures()
    paired = paired_figures()
    detection_examples()
    failure_examples()
    alpha = alpha_figure()
    efficiency, environment = efficiency_figures()
    write_documents(thresholds, paired, alpha, efficiency, environment)
    print(f"Generated {len(FIGURES)} figures and {len(TABLES)} tables: {OUT}")


if __name__ == "__main__":
    main()
