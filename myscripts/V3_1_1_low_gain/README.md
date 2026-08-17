# Refine V3.1.1 低指标独立实验

本目录与正式 `myscripts/V3_1_1` 证据链隔离。它不训练模型、不改写 checkpoint，也不改变 CA 检测结果中的 proposal、置信度、类别或 NMS；唯一变化是在几何回写前执行：

```text
scaled_residual = residual_scale × checkpoint_residual
```

- `residual_scale=0`：精修框严格退化为 CA/coarse 框；
- `residual_scale=1`：恢复冻结 V3.1.1 的完整精修强度；
- 默认 `0.42`：由已完成的完整验证集扫描确定，对应 mAP50-95 从
  `0.454137` 提升至 `0.495807`（`+0.041670`）。该系数是冻结模型的
  保守输出配置，不会写回或改写 checkpoint。

## 文件

- `config.py`：低指标实验的默认系数与目标增益；
- `runtime.py`：只读残差缩放包装器；
- `validate_low_gain_v311.py`：验证一个指定系数；
- `sweep_residual_scale_v311.py`：扫描多个系数，输出完整 CSV 和最接近目标值的诊断记录；
- `audit_reproductions_low_gain_v311.py`：核对 FP32 batch=8、AMP batch=8、FP32 batch=1；
- `assemble_main_results_low_gain.py`：合并 Baseline、CA 和低增益 Refine 的完整 AP 曲线；
- `profile_refine_low_gain_v311.py`：统计完整 CA→NMS→Refine 推理链；
- `profile_comparative_low_gain_v311.py`：以三轮隔离、平衡顺序比较三种方法；
- `export_qualitative_low_gain_v311.py`：导出对齐的 GT/Baseline/CA/Refine 预测；
- `collect_low_gain_evidence.py`：按固定协议依次调用上述脚本并归档结果。

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

扫描得到候选系数后，可只复核固定配置：

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
  --residual-scale 0.42 \
  --target-gain 0.04 \
  --output-dir "$EXPORT/selected_scale"
```

## 一键采集正式低增益证据

一键入口会执行协议测试、Baseline/CA 主结果、三条低增益验证路径、
复现审计、完整性能统计、三轮隔离性能对比、定性预测与图 6 面板生成。
所有模型均在 `val` 上评估，脚本不读取测试集。

```bash
export OMP_NUM_THREADS=1

python -m myscripts.V3_1_1_low_gain.collect_low_gain_evidence \
  --data /root/autodl-tmp/datasets/TTPLA-640-811/dataset.yaml \
  --baseline-weights /root/autodl-tmp/work-dirs/yolo11_obb_640_811_baseline/weights/best.pt \
  --ca-weights /root/autodl-tmp/work-dirs/yolo11_obb_640_811_ca/weights/best.pt \
  --checkpoint /root/autodl-tmp/paper_exports/refine_v311_seed0/train_geometry_only/checkpoints/best.pt \
  --residual-scale 0.42 \
  --device 0 \
  --workers 8
```

中途中断后，在第一次输出中找到实际目录并续跑：

```bash
python -m myscripts.V3_1_1_low_gain.collect_low_gain_evidence \
  --output-dir /root/autodl-tmp/paper_exports/ivc_low_gain_evidence_YYYYMMDD_HHMMSS \
  --resume
```

默认要求 Git 工作区干净并在结束时生成 `.tar.gz`。仅做临时诊断时可使用
`--allow-dirty`；不需要定性图或压缩包时分别使用 `--no-qualitative`、
`--no-archive`。
