# Refine V3.1 训练与验收说明

## 1. 固定实验边界

- 使用原有数据划分，不重新划分数据集。
- CA 权重固定为
  `/root/autodl-tmp/work-dirs/yolo11_obb_640_811_ca/weights/best.pt`。
- 图像尺寸固定为 640。
- CA 始终处于 `eval()` 且参数冻结；优化器只接收 V3.1 参数。
- V3.1 对全部 CA 后 NMS proposal 进行一次精修，不使用推理质量门控，不执行第二次 NMS。
- 只预测短边和长边对数尺度残差，不修改中心、角度、类别和置信度。
- test 在结构与训练规则冻结前保持未使用。

## 2. 两个等预算实验

`geometry_only` 是优先版本。它没有 quality head，checkpoint 中也不包含 quality 参数。

`quality_aux` 使用相同几何结构和训练预算，额外加入训练期 quality 辅助损失。quality 输出不参与推理。

两个实验的默认尺度边界固定为：

| 输出 | 对数残差范围 | 对应尺度范围 |
|---|---:|---:|
| 短边 | `[-0.50, +0.20]` | `[0.607, 1.221]` |
| 长边 | `[-0.08, +0.08]` | `[0.923, 1.083]` |

## 3. 训练命令

```bash
export OMP_NUM_THREADS=1

BASE=/root/autodl-tmp/work-dirs/yolo11_obb_640_811_ca/weights/best.pt
DATA=/root/autodl-tmp/datasets/TTPLA-640-811/dataset.yaml
EXPORT=/root/autodl-tmp/paper_exports/refine_v31_seed0

python -m myscripts.V3_1.smoke_refine_v31 \
  --experiment geometry_only \
  --ca-weights "$BASE" \
  --data "$DATA" \
  --imgsz 640 \
  --batch 2 \
  --device 0 \
  --workers 2 \
  --output "$EXPORT/smoke_geometry_only.json"

python -m myscripts.V3_1.smoke_refine_v31 \
  --experiment quality_aux \
  --ca-weights "$BASE" \
  --data "$DATA" \
  --imgsz 640 \
  --batch 2 \
  --device 0 \
  --workers 2 \
  --output "$EXPORT/smoke_quality_aux.json"

python -m myscripts.V3_1.train_refine_v31 \
  --experiment geometry_only \
  --ca-weights "$BASE" \
  --data "$DATA" \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --epochs 30 \
  --eval-interval 5 \
  --seed 0 \
  --holdout-fraction 0.20 \
  --output-dir "$EXPORT/geometry_only"

python -m myscripts.V3_1.train_refine_v31 \
  --experiment quality_aux \
  --ca-weights "$BASE" \
  --data "$DATA" \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --epochs 30 \
  --eval-interval 5 \
  --seed 0 \
  --holdout-fraction 0.20 \
  --output-dir "$EXPORT/quality_aux"
```

两个命令独立运行，均从同一纯 CA 权重开始。第二个实验不接续第一个实验的 V3.1 权重。

## 4. 独立复现命令

训练脚本会自动在 val 上运行一次。下载结果前，再用选中的 `best.pt` 独立复现：

```bash
python -m myscripts.V3_1.validate_refine_v31 \
  --checkpoint "$EXPORT/geometry_only/checkpoints/best.pt" \
  --ca-weights "$BASE" \
  --data "$DATA" \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --no-amp \
  --output-dir "$EXPORT/geometry_only_reproduction"

python -m myscripts.V3_1.validate_refine_v31 \
  --checkpoint "$EXPORT/quality_aux/checkpoints/best.pt" \
  --ca-weights "$BASE" \
  --data "$DATA" \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --no-amp \
  --output-dir "$EXPORT/quality_aux_reproduction"

python -m myscripts.V3_1.summarize_v31_training \
  --geometry-dir "$EXPORT/geometry_only" \
  --quality-aux-dir "$EXPORT/quality_aux" \
  --noninferiority-tolerance 0.002 \
  --output-dir "$EXPORT/decision"
```

## 5. 输出目录

每个训练目录包括：

| 文件 | 内容 |
|---|---|
| `run_manifest.json` | 数据、CA 哈希、结构、超参数和推理策略 |
| `train_history.csv` | 每轮损失、匹配率、目标裁剪率和学习率 |
| `holdout_metrics.csv` | 每 5 轮的 coarse/identity/refined 指标 |
| `holdout_diagnostics_epoch*.json` | 残差、边界占用和匹配 IoU 诊断 |
| `checkpoints/epoch*.pt` | 定期 checkpoint |
| `checkpoints/best.pt` | 仅按 holdout mAP、AP75、AP90选择的权重 |
| `selection.json` | 固定检查点选择依据 |
| `val_metrics.csv` | val 上唯一一次完整指标 |
| `val_diagnostics.json` | val 残差与匹配 proposal 统计 |
| `acceptance.json` | 预声明验收结果 |
| `RESULTS.md` | 人工快速核对摘要 |

## 6. 验收规则

- coarse 与登记 CA mAP50-95 的差值不超过 `0.002`；
- identity 与 coarse 的 mAP50-95、AP75、AP90、AP95 最大差值不超过 `5e-4`；
- V3.1 相对 CA 的 mAP50-95 提高至少 `0.03`；
- AP75 不下降，AP90 下降不超过 `0.002`；
- 匹配 proposal 的平均 IoU 改变量非负，改善比例不低于恶化比例；
- 短边、长边接近输出边界的比例均不高于 `10%`；
- 残差不是常数；
- CA 文件训练前后 SHA256 完全一致。

`+0.03～+0.05` 是保守目标，不是成绩上限。如果提升更高，先执行真实性审计，不人为压低指标。
