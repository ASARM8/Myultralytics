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
| `V3_IMPLEMENTATION_REVIEW.md` | 数据结论、设计依据、代码审核、风险边界和验收标准 |
| `ultralytics/nn/modules/refine_v3/proposal_refine.py` | P2/P3 旋转条带编码和尺度/质量预测模块 |

## 3. 云端回归测试

```bash
export OMP_NUM_THREADS=1
BASE=/root/autodl-tmp/work-dirs/yolo11_obb_640_811_ca/weights/best.pt
DATA=/root/autodl-tmp/datasets/TTPLA-640-811/dataset.yaml
EXPORT=/root/autodl-tmp/paper_exports/refine_v3

python -m pytest -q \
  tests/V3/test_refine_proposal_oracle.py \
  tests/V3/test_analyze_obb_resolution.py \
  tests/V3/test_refine_rotated_roi_probe.py \
  tests/V3/test_proposal_refine_v3.py \
  tests/V3/test_runtime_v3.py \
  tests/V3/test_train_refine_v3.py
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
EXPORT=/root/autodl-tmp/paper_exports/refine_v3

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
  --output-dir "$EXPORT/proposal_oracle_corrected"
```

首先检查：

- `standard_ca` 是否落在 `0.45413±0.002`；否则脚本会在产出诊断文件后报错，不应启动正式训练。
- `postnms_coarse` 是否与 `standard_ca` 基本一致。
- `postnms_oracle_scale` 是否相对 `postnms_coarse` 有明确正上限。它比 pre-NMS oracle 更贴近 V3 的实际作用位置。

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
  --lr 3e-4 \
  --warmup-epochs 3 \
  --weight-decay 1e-4 \
  --expected-ca-map50-95 0.45413 \
  --output-dir "$EXPORT/train_seed0"
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
  --checkpoint "$EXPORT/train_seed0/checkpoints/best.pt" \
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

## 7. 当前训练定义

- 处理 CA 的 post-NMS proposal，保留类别和置信度。
- 使用 P2/P3 旋转对齐条带及 proposal 状态特征。
- 首版只预测 `[δ_short, δ_long]`；中心通道固定为 0。
- 残差范围为短边 `[-1.5, 0.25]`、长边 `[-0.15, 0.15]`，使用符号相关 `tanh`，目标限制在边界的 99% 以内。
- matched proposal 使用尺度 SmoothL1；短边小于 8 px 的样本平滑降权但不删除。
- 独立质量分支判断“有界尺度 oracle 是否至少改善 ProbIoU 0.002”，推理时只门控几何修正，不改分类分数。
- 质量阴性或未匹配 proposal 加轻量 identity 约束；精修后重新执行旋转 NMS。

所有数值仍属于 seed0 筛选设计，不应在完成多种子配对实验前写成正式论文结论。
