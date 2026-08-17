# Refine V3.1.1 低指标独立实验

本目录与正式 `myscripts/V3_1_1` 证据链隔离。它不训练模型、不改写 checkpoint，也不改变 CA 检测结果中的 proposal、置信度、类别或 NMS；唯一变化是在几何回写前执行：

```text
scaled_residual = residual_scale × checkpoint_residual
```

- `residual_scale=0`：精修框严格退化为 CA/coarse 框；
- `residual_scale=1`：恢复冻结 V3.1.1 的完整精修强度；
- 默认 `0.37`：根据现有 `+0.108368` 增益对目标 `+0.04` 的线性初始估计，不代表最终实测值。

## 文件

- `config.py`：低指标实验的默认系数与目标增益；
- `validate_low_gain_v311.py`：验证一个指定系数；
- `sweep_residual_scale_v311.py`：扫描多个系数，输出完整 CSV 和最接近目标值的诊断记录。

## 正式使用边界

该目录默认只做敏感性分析，不能因为某个结果更接近期望的论文数字而隐藏 `residual_scale=1` 的完整结果。若将低强度版本作为部署配置，应先声明独立的稳健性选择条件，例如在 AP95 不下降的约束下选择验证集 mAP50-95 最高的系数，并报告完整扫描结果。

运行命令见项目的图表与数据复现说明，或直接执行：

```bash
export OMP_NUM_THREADS=1

BASE=/root/autodl-tmp/work-dirs/yolo11_obb_640_811_ca/weights/best.pt
DATA=/root/autodl-tmp/datasets/TTPLA-640-811/dataset.yaml
REFINE=/root/autodl-tmp/paper_exports/refine_v311_seed0/train_geometry_only/checkpoints/best.pt
EXPORT=/root/autodl-tmp/paper_exports/refine_v311_low_gain

python -m myscripts.V3_1_1_low_gain.sweep_residual_scale_v311 \
  --checkpoint "$REFINE" \
  --ca-weights "$BASE" \
  --data "$DATA" \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --no-amp \
  --scales "0,0.20,0.28,0.32,0.35,0.37,0.39,0.42,0.45,0.50,0.75,1.0" \
  --target-gain 0.04 \
  --output-dir "$EXPORT"
```

扫描得到候选系数后，可只复核该系数：

```bash
python -m myscripts.V3_1_1_low_gain.validate_low_gain_v311 \
  --checkpoint "$REFINE" \
  --ca-weights "$BASE" \
  --data "$DATA" \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --no-amp \
  --residual-scale 0.37 \
  --target-gain 0.04 \
  --output-dir "$EXPORT/selected_scale"
```
