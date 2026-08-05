# Refine V3：候选框级旋转 ROI 精修

## 1. 固定边界

- 纯 CA 权重固定为：

  ```text
  /root/autodl-tmp/work-dirs/yolo11_obb_640_811_ca/weights/best.pt
  ```

- 输入固定为 `imgsz=640`。
- V3 与旧 `OBBRefine`、`OBBRefineV2` 完全隔离。
- 设计和阈值冻结前禁止读取 test；现有 V3 脚本只开放 train/val。
- CA 全部参数和 BatchNorm 状态保持冻结；V3 只训练独立 proposal refiner。

## 2. 文件与作用

| 文件 | 作用 |
|---|---|
| `refine_proposal_oracle.py` | D1：pre-NMS 与 post-NMS proposal 几何 oracle、proposal recall、CA 基线校验 |
| `refine_rotated_roi_probe.py` | D2：比较状态特征与 P2/P3 旋转 ROI 的可学习性 |
| `analyze_obb_resolution.py` | D3：640 输入下 OBB 尺度及 1 像素扰动敏感性 |
| `runtime.py` | 冻结 CA 特征/候选框提取、匹配、监督构造、旋转 NMS 和指标计算 |
| `smoke_refine_v3.py` | 只读取一个 train batch，检查真实数据上的特征 hook、identity、匹配、反向传播和 CA 无梯度 |
| `train_refine_v3.py` | 正式 seed0 训练；train-fit/holdout 选择 epoch 与质量阈值，随后只评一次 val |
| `validate_refine_v3.py` | 复现冻结 checkpoint；校验 CA 文件哈希、baseline 和 NMS roundtrip identity |
| `audit_refine_v3.py` | 冻结 checkpoint 的完整真实性审计：门控、均值残差、shuffle、短/长边、re-NMS、匹配 IoU 与分组统计 |
| `audit_dataset_splits_v3.py` | train/val 路径、文件 SHA256、dHash 近重复与人工拼图审计；不读取 test |
| `V3_IMPLEMENTATION_REVIEW.md` | 数据结论、设计依据、代码审核、风险边界和验收标准 |
| `ultralytics/nn/modules/refine_v3/proposal_refine.py` | P2/P3 旋转条带编码和尺度/质量预测模块 |

## 3. 云端回归测试

```bash
export OMP_NUM_THREADS=1
BASE=/root/autodl-tmp/work-dirs/yolo11_obb_640_811_ca/weights/best.pt
DATA=/root/autodl-tmp/datasets/TTPLA-640-811/dataset.yaml
EXPORT=/root/autodl-tmp/paper_exports/refine_v3_seed0_01

python -m pytest -q \
  tests/V3/test_refine_proposal_oracle.py \
  tests/V3/test_analyze_obb_resolution.py \
  tests/V3/test_refine_rotated_roi_probe.py \
  tests/V3/test_proposal_refine_v3.py \
  tests/V3/test_runtime_v3.py \
  tests/V3/test_train_refine_v3.py \
  tests/V3/test_audit_refine_v3.py \
  tests/V3/test_audit_dataset_splits_v3.py
```

测试覆盖：参数边界、GT 加权 recall、分桶、分组无泄漏、类别感知一对一匹配、OBB 等价表示、零初始化 identity、非对称残差边界和 checkpoint 选择规则。

随后运行一个真实 train batch 的集成检查（不读取 val/test）：

```bash
python -m myscripts.V3.smoke_refine_v3 \
  --ca-weights "$BASE" \
  --data "$DATA" \
  --imgsz 640 \
  --batch 2 \
  --device 0 \
  --workers 2 \
  --output "$EXPORT/smoke_train_batch.json"
```

## 4. D1 修正版：冒烟检查后运行

旧 D1 显式使用 `proposal_conf=0.001`，而当前仓库 OBB 验证默认使用 `0.01`，因此得到的 standard CA mAP50-95=0.421885 不能与训练记录的 0.45413 直接比较。修正版统一为 `0.01`、增加预期基线硬检查，并同时输出 V3 实际处理对象（post-NMS proposal）的 oracle。

```bash
BASE=/root/autodl-tmp/work-dirs/yolo11_obb_640_811_ca/weights/best.pt
DATA=/root/autodl-tmp/datasets/TTPLA-640-811/dataset.yaml
EXPORT=/root/autodl-tmp/paper_exports/refine_v3_seed0_01

python -m myscripts.V3.refine_proposal_oracle \
  --weights "$BASE" \
  --data "$DATA" \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --split val \
  --proposal-topk 1000 \
  --proposal-conf 0.01 \
  --oracle-match-iou 0.30 \
  --nms-iou 0.70 \
  --max-det 300 \
  --expected-ca-map50-95 0.45413 \
  --baseline-tolerance 0.002 \
  --roundtrip-tolerance 0.0005 \
  --output-dir "$EXPORT/proposal_oracle_corrected_v2"
```

首先检查：

- `standard_ca` 是否落在 `0.45413±0.002`；否则脚本会在产出诊断文件后报错，不应启动正式训练。
- `postnms_coarse` 与 `standard_ca` 的 mAP50-95 绝对差是否不超过 `5e-4`；否则脚本会在保存诊断后报错。
- `postnms_oracle_scale` 是否相对 `postnms_coarse` 有明确正上限。它比 pre-NMS oracle 更贴近 V3 的实际作用位置。

当前 D1 v2 已完成，结果为：standard CA=0.4541379，基线绝对误差约 `7.9e-6`；top-K coarse 与 standard CA 六项指标完全一致；post-NMS roundtrip 误差为 0；post-NMS scale oracle=0.787313（相对 coarse +0.333175）。三项训练前硬检查均已通过。oracle 仅表示几何上限，不是模型实际成绩。

## 5. seed0 正式训练

```bash
python -m myscripts.V3.train_refine_v3 \
  --ca-weights "$BASE" \
  --data "$DATA" \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --epochs 30 \
  --eval-interval 5 \
  --seed 0 \
  --proposal-conf 0.01 \
  --nms-iou 0.70 \
  --max-det 300 \
  --match-iou 0.30 \
  --quality-min-gain 0.002 \
  --quality-thresholds 0.3,0.5,0.7,0.9 \
  --roi-height 5 \
  --roi-width 24 \
  --roi-channels 32 \
  --hidden-channels 128 \
  --short-negative-limit 1.5 \
  --short-positive-limit 0.25 \
  --long-negative-limit 0.15 \
  --long-positive-limit 0.15 \
  --target-margin 0.99 \
  --lr 3e-4 \
  --warmup-epochs 3 \
  --weight-decay 1e-4 \
  --expected-ca-map50-95 0.45413 \
  --output-dir "$EXPORT/train_seed0_scale_only"
```

若文件名能识别同一场景的连续帧，应在命令中增加类似：

```bash
  --group-regex '(scene[0-9]+)_'
```

默认按完整图像路径分组。训练阶段不读取 val；每 5 轮只在 train-holdout 比较质量阈值并选择 checkpoint。epoch 与阈值冻结后，脚本才构造 val 数据并评估一次。test 始终不使用。

主要输出：

- `checkpoints/best.pt`、`last.pt`、`epochXXX.pt`
- `train_history.csv`
- `holdout_metrics.csv` 与逐检查点诊断 JSON
- `selection.json`
- `val_metrics.csv`、`val_diagnostics.json`
- `acceptance.json`、`training_report.md`

训练入口若发现输出目录已有 `run_manifest.json` 会拒绝覆盖，请为重跑使用新目录，避免不同实验文件混在一起。

## 6. 独立复现验证

训练脚本已自动进行一次冻结后的 val。只有需要复现实验或审核 checkpoint 时，再运行独立验证；不要用它反复调阈值。

```bash
python -m myscripts.V3.validate_refine_v3 \
  --checkpoint "$EXPORT/train_seed0_scale_only/checkpoints/best.pt" \
  --ca-weights "$BASE" \
  --data "$DATA" \
  --split val \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --expected-ca-map50-95 0.45413 \
  --output-dir "$EXPORT/validate_seed0_reproduction"
```

验证器默认从 checkpoint 读取 train-holdout 已选阈值，并拒绝 CA 文件哈希不一致、非纯 CA 头、非 `reg_max=32`、baseline 漂移或 post-NMS roundtrip 不恒等。

当前 seed0 已完成独立复现：AMP/batch=8 的 `val_metrics.csv` 与训练后首次 val 逐字节一致，mAP50-95 为 0.695961。FP32/batch=1 得到 coarse=0.454151、refined=0.698983、Δ=+0.244832，方向和幅度稳定，但说明半精度与批大小需要拆开记录，正式审计固定使用 FP32。

## 7. 完整真实性审计

### 7.1 先拆分 batch 与精度影响

以下两次只用于数值稳定性审核，不重新选择 checkpoint 或 threshold：

```bash
python -m myscripts.V3.validate_refine_v3 \
  --checkpoint "$EXPORT/train_seed0_scale_only/checkpoints/best.pt" \
  --ca-weights "$BASE" \
  --data "$DATA" \
  --split val \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --no-amp \
  --expected-ca-map50-95 0.45413 \
  --output-dir "$EXPORT/validate_seed0_fp32_batch8"

python -m myscripts.V3.validate_refine_v3 \
  --checkpoint "$EXPORT/train_seed0_scale_only/checkpoints/best.pt" \
  --ca-weights "$BASE" \
  --data "$DATA" \
  --split val \
  --imgsz 640 \
  --batch 1 \
  --device 0 \
  --workers 8 \
  --amp \
  --expected-ca-map50-95 0.45413 \
  --output-dir "$EXPORT/validate_seed0_amp_batch1"
```

与已有 AMP/batch=8、FP32/batch=1 组成 2×2 对照。论文正式口径采用 FP32；batch 变化不应改变结论方向，且同精度下 mAP50-95 建议相差不超过 `0.002`。

### 7.2 机制真实性审计

```bash
python -m myscripts.V3.audit_refine_v3 \
  --checkpoint "$EXPORT/train_seed0_scale_only/checkpoints/best.pt" \
  --ca-weights "$BASE" \
  --data "$DATA" \
  --split val \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --no-amp \
  --shuffle-seed 20250301 \
  --expected-ca-map50-95 0.45413 \
  --expected-refined-map50-95 0.69898 \
  --refined-tolerance 0.002 \
  --output-dir "$EXPORT/truth_audit_fp32_batch8"
```

脚本从 checkpoint 读取 epoch15 和 train-holdout 已选 threshold=0.3，不允许在 val 重选。它输出：

- `mechanism_metrics.csv`：coarse、roundtrip、gate-off、正常门控、全开、短/长边、冻结均值残差、residual/quality/spatial shuffle、去除 re-NMS；
- `matched_proposal_diagnostics.csv`：逐匹配 proposal 的预测残差、quality、coarse/refined/oracle IoU；
- `quality_audit.json`：quality 对真实学习收益和 bounded oracle 收益的 AUC、Brier、精确率/召回率及相关性；
- `subgroup_metrics.csv`：按短边、长宽比和置信度的改善/恶化比例；
- `truth_audit.json` 与 `truth_audit_report.md`：硬完整性检查和各机制差值。

`gate_off`、`roundtrip` 必须与 coarse 恒等。若正常结果不能明显优于均值残差或 residual shuffle，整体增益仍可是真实的，但论文不能把它解释为实例级 ROI 精修；若 quality shuffle 与正常结果接近，则 quality gate 的独立贡献不足。

### 7.3 train/val 数据划分审计

```bash
python -m myscripts.V3.audit_dataset_splits_v3 \
  --data "$DATA" \
  --imgsz 640 \
  --near-hamming 4 \
  --max-near-pairs 5000 \
  --contact-sheet-pairs 40 \
  --output-dir "$EXPORT/dataset_split_audit"
```

完整路径或文件 SHA256 跨 train/val 重复会在保存证据后报错。dHash 近重复只作为人工核对候选，脚本会生成 `near_duplicate_contact_sheet_*.jpg`，不能仅凭感知哈希自动判为泄漏。test 全程不读取。

## 8. 当前训练定义

- 处理 CA 的 post-NMS proposal，保留类别和置信度。
- 使用 P2/P3 旋转对齐条带及 proposal 状态特征。
- 首版只预测 `[δ_short, δ_long]`；中心通道固定为 0。
- 残差范围为短边 `[-1.5, 0.25]`、长边 `[-0.15, 0.15]`，使用符号相关 `tanh`，目标限制在边界的 99% 以内。
- matched proposal 使用尺度 SmoothL1；短边小于 8 px 的样本平滑降权但不删除。
- 独立质量分支判断“有界尺度 oracle 是否至少改善 ProbIoU 0.002”，推理时只门控几何修正，不改分类分数。
- 质量阴性或未匹配 proposal 加轻量 identity 约束；精修后重新执行旋转 NMS。

所有数值仍属于 seed0 筛选设计，不应在完成多种子配对实验前写成正式论文结论。

## 9. V3 已完成审计与 V3.1 入口

V3 的独立复现、FP32/AMP 与 batch 对照、完整机制真实性审计和 train/val 划分审计均已完成。当前证据为：

- proposal 对应 residual 和旋转 ROI 空间特征均有真实贡献；
- 仅短边残差基本复现完整收益；
- 当前 quality gate 略低于 all-refine；
- selected 路径的二次 NMS 不改变指标；
- 数据中存在 13 对精确 train/val 重复，涉及 7 张唯一 val 图像。

下一步不在本目录继续扩张 V3，而是运行独立的冻结前验证：

```text
myscripts/V3_1/README.md
```

该验证会在 train-holdout 和 clean-val 上用同一组预声明规则判断 V3.1 是否删除长边分支、质量门控与二次 NMS。完整历史记录见 `mydocs/创新点一/Refine_V1至V3.1工作记录.md`。
