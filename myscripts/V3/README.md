# Refine V3 第一阶段：候选框与旋转 ROI 诊断

## 1. 当前边界

- 旧 V1/V2.x 的失败只说明“逐位置、双通道 FPN 残差头”没有学到有效的实例修正，不等价于所有 Refine 方案都不可行。
- 第一阶段固定使用纯 CA 权重：

  ```text
  /root/autodl-tmp/work-dirs/yolo11_obb_640_811_ca/weights/best.pt
  ```

- 输入固定为 `imgsz=640`。
- 结构、阈值和训练方案冻结前禁止读取 test；本阶段脚本只开放 train/val。
- 所有诊断结果用于确定 V3 的监督对象、特征范围和安全残差范围，不自动生成“停止 Refine”的决定。

## 2. 已实现内容

| 内容 | 文件 | 作用 |
|---|---|---|
| D1 proposal oracle | `myscripts/V3/refine_proposal_oracle.py` | 在真实 pre-NMS 候选上修改几何并重新执行旋转 NMS，输出完整 mAP/AP75/AP90/AP95 |
| D2 rotated ROI probe | `myscripts/V3/refine_rotated_roi_probe.py` | 比较 proposal 状态 MLP 与 P2/P3 旋转对齐条带特征，按图像/场景分组拆分 train-fit/train-holdout |
| D3 resolution audit | `myscripts/V3/analyze_obb_resolution.py` | 统计 640 输入下目标短边、长边、长宽比及 1 像素几何扰动的 `1-ProbIoU` |
| V3 独立模块 | `ultralytics/nn/modules/refine_v3/proposal_refine.py` | proposal 级 P2/P3 旋转条带编码，预测 scale+center 残差和独立质量 logit |

V3 模块与 `OBBRefine`、`OBBRefineV2` 完全隔离。几何输出层零初始化，因此新模块初始时是严格 identity；它不修改 CA 分类分数或类别。

## 3. 云端回归测试

```bash
export OMP_NUM_THREADS=1

pytest -q \
  tests/V3/test_refine_proposal_oracle.py \
  tests/V3/test_analyze_obb_resolution.py \
  tests/V3/test_refine_rotated_roi_probe.py \
  tests/V3/test_proposal_refine_v3.py
```

验收点：参数解析、GT 加权 proposal recall、分桶边界、分组无泄漏，以及 V3 零初始化 identity 均通过。

## 4. D1：真实 proposal oracle

```bash
BASE=/root/autodl-tmp/work-dirs/yolo11_obb_640_811_ca/weights/best.pt
DATA=/root/autodl-tmp/datasets/TTPLA-640-811/dataset.yaml
EXPORT=/root/autodl-tmp/paper_exports/refine_v3_stage1

python -m myscripts.V3.refine_proposal_oracle \
  --weights "$BASE" \
  --data "$DATA" \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --split val \
  --proposal-topk 1000 \
  --proposal-conf 0.001 \
  --oracle-match-iou 0.30 \
  --nms-iou 0.70 \
  --max-det 300 \
  --output-dir "$EXPORT/proposal_oracle"
```

主要输出：

- `proposal_oracle_metrics.csv`：standard CA、top-K coarse、scale、center、angle、scale+center 和完整几何 oracle 的检测指标。
- `proposal_recall_summary.csv`：IoU=0.50/0.75/0.90 的 GT 加权 proposal recall。
- `proposal_recall_by_image.csv`：逐图像 recall，便于定位候选框缺失场景。
- `proposal_oracle_report.md`：汇总与解释边界。

先检查 `topk_coarse` 与 `standard_ca`。若 mAP50-95 差值绝对值大于 `5e-4`，应先增加 `--proposal-topk`，不能直接解释其他 oracle。

## 5. D3：640 分辨率审计

分别统计无增强的 train 和 val：

```bash
python -m myscripts.V3.analyze_obb_resolution \
  --data "$DATA" \
  --split train \
  --imgsz 640 \
  --batch 16 \
  --workers 8 \
  --output-dir "$EXPORT/resolution_train"

python -m myscripts.V3.analyze_obb_resolution \
  --data "$DATA" \
  --split val \
  --imgsz 640 \
  --batch 16 \
  --workers 8 \
  --output-dir "$EXPORT/resolution_val"
```

主要输出：

- `obb_resolution_instances.csv`：逐实例像素尺度和七种 1 像素扰动敏感度。
- `obb_resolution_summary.csv`：按短边、长宽比和类别分组的统计。
- `obb_resolution_report.md`：tiny 比例、尺度分位数和短边分桶敏感度。

若大量目标短边小于 4 像素，且 1 像素短轴扰动造成较大的 `1-ProbIoU`，V3 应限制残差幅度，并避免把标注离散噪声当作可精修信号；这不是放弃 Refine，而是改变监督与分组策略。

## 6. D2：旋转 ROI 可学习性

```bash
python -m myscripts.V3.refine_rotated_roi_probe \
  --ca-weights "$BASE" \
  --data "$DATA" \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --train-split train \
  --eval-split val \
  --max-train-samples 20000 \
  --max-eval-samples 12000 \
  --match-iou 0.30 \
  --roi-height 5 \
  --roi-width 24 \
  --projection-channels 16 \
  --epochs 50 \
  --output-dir "$EXPORT/rotated_roi_probe"
```

如数据文件名含同一场景的连续帧，应增加场景提取规则，例如：

```bash
  --group-regex '(scene[0-9]+)_'
```

正则的第一个捕获组会作为 scene group；同组样本不会跨 train-fit/train-holdout。若没有连续场景信息，默认按完整图像路径分组，至少消除同一图像/同一 GT 的 anchor 级泄漏。

主要输出：

- `probe_metrics.csv`：state/ROI 在 train-fit、train-holdout、val 上的 scale、center、combined MAE、方向准确率和 ProbIoU 变化。
- `target_distribution.csv`：四个 V3 目标的分布与截断比例。
- `state_probe.pt`、`roi_probe.pt`：复核用 probe 权重，不是正式检测模型权重。
- `rotated_roi_probe_report.md`：按预先定义的证据边界汇总。

解释顺序：

1. 先看 train-fit 与 train-holdout 的差距，判断是否只是记忆训练图像。
2. 再看 val；只有 holdout 和 val 同方向，才能进入正式训练。
3. 比较 ROI 与 state。ROI 稳定更好，说明空间对齐特征提供了原 pointwise head 缺失的信息。
4. 若 ROI 仍弱，下一轮调整 proposal 匹配、条带上下文、P2/P3 投影和监督分组；不得自动写成“Refine 不可行”。

## 7. 后续正式训练

`OBBProposalRefinerV3` 的网络模块已经实现，但正式 trainer/validator 暂不在第一阶段启用。原因是 D1-D3 将确定以下内容：

- 是否先训练 `scale+center`，还是 scale/center 分阶段训练；
- proposal top-K 和最低匹配 IoU；
- P2/P3 条带宽度及短轴最小上下文；
- scale、center 的残差安全范围；
- 质量分支使用的正负监督和阈值。

获得第一阶段结果后再冻结这些定义，并实现 proposal 级 trainer、rotated NMS 前后验证和 CA identity 回归。无论 D2 初版结果是否通过，下一步都是针对证据修改 V3，而不是自动终止 Refine 研究。
