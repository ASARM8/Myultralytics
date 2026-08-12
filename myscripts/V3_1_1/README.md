# Refine V3.1.1：平滑保守监督

## 1. 单因素修改

V3.1.1 只修改 V3.1 的监督目标映射，不修改 CA、旋转 ROI、特征通道、proposal 策略或推理解码：

- 固定 `geometry_only`，无 quality head；
- 物理输出边界保持短边 `[-0.50,+0.20]`、长边 `[-0.08,+0.08]`；
- 监督范围固定为物理边界的 80%；
- 使用分符号 `tanh` 平滑压缩，不再把超界目标硬裁剪为同一个常数；
- 训练 15 轮，每轮在原 image-level holdout 上评估；
- 推理精修全部 proposal，不使用质量门控，不执行第二次 NMS；
- 新增 AP95 下降不超过 `0.002` 的验收规则。

短边负目标的变换示意为：

\[
t'_s=-0.40\tanh\left(\frac{-t_s}{0.40}\right),\qquad t_s<0.
\]

该映射在零点附近导数为 1，极端目标渐近于 `-0.40`，但不会形成硬裁剪产生的边界点质量堆积。

### 合理性与风险审核

- 这是单因素试验：推理侧允许的最大修正量不变，因此可直接判断变化是否来自监督目标映射，而不是更换 proposal、ROI 或解码规则；
- 零点附近斜率为 1，小残差样本不会因压缩而失去一阶监督信号；
- 监督目标与物理边界之间保留 20% 余量，针对 V3.1 短边输出堆积在边界的问题；
- 平滑压缩会弱化极端样本的目标幅度，可能降低 AP95，因此 AP95 被加入硬验收条件，而不是只观察 mAP50-95；
- 80% 是根据 V3.1 已观测边界堆积预先固定的单次设置，本轮不进行多比例搜索，避免继续在 val 上后验调参；
- 若预测输出仍大量到达物理边界，说明问题不只来自硬裁剪，`short_boundary_pass` 会失败并保留诊断证据。

## 2. 云端命令

```bash
export OMP_NUM_THREADS=1

BASE=/root/autodl-tmp/work-dirs/yolo11_obb_640_811_ca/weights/best.pt
DATA=/root/autodl-tmp/datasets/TTPLA-640-811/dataset.yaml
EXPORT=/root/autodl-tmp/paper_exports/refine_v311_seed0

python -m pytest -q tests/V3 tests/V3_1 tests/V3_1_1

python -m myscripts.V3_1_1.smoke_refine_v311 \
  --ca-weights "$BASE" \
  --data "$DATA" \
  --imgsz 640 \
  --batch 2 \
  --device 0 \
  --workers 2 \
  --output "$EXPORT/smoke_geometry_only.json"

python -m myscripts.V3_1_1.train_refine_v311 \
  --ca-weights "$BASE" \
  --data "$DATA" \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --output-dir "$EXPORT/train_geometry_only"
```

入口会自动锁定原 V3.1 的 seed0、20% image-level holdout 和全部模型/优化参数，并固定 `geometry_only`、监督比例 `0.80`、训练轮数 `15`、评估间隔 `1` 和 AP95 最大下降 `0.002`。命令行只需提供数据、设备、batch、workers、AMP模式和输出目录。

## 3. 独立复现

先使用与训练相同的 AMP、batch=8 复现，再用 FP32 检查数值边界：

```bash
python -m myscripts.V3_1_1.validate_refine_v311 \
  --checkpoint "$EXPORT/train_geometry_only/checkpoints/best.pt" \
  --ca-weights "$BASE" \
  --data "$DATA" \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --output-dir "$EXPORT/reproduction_amp_batch8"

python -m myscripts.V3_1_1.validate_refine_v311 \
  --checkpoint "$EXPORT/train_geometry_only/checkpoints/best.pt" \
  --ca-weights "$BASE" \
  --data "$DATA" \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --no-amp \
  --output-dir "$EXPORT/reproduction_fp32_batch8"

python -m myscripts.V3_1_1.validate_refine_v311 \
  --checkpoint "$EXPORT/train_geometry_only/checkpoints/best.pt" \
  --ca-weights "$BASE" \
  --data "$DATA" \
  --imgsz 640 \
  --batch 1 \
  --device 0 \
  --workers 8 \
  --no-amp \
  --output-dir "$EXPORT/reproduction_fp32_batch1"

python -m myscripts.V3_1_1.audit_reproductions_v311 \
  --training-dir "$EXPORT/train_geometry_only" \
  --amp-batch8-dir "$EXPORT/reproduction_amp_batch8" \
  --fp32-batch8-dir "$EXPORT/reproduction_fp32_batch8" \
  --fp32-batch1-dir "$EXPORT/reproduction_fp32_batch1" \
  --output-dir "$EXPORT/reproduction_audit"
```

## 4. 验收重点

- `acceptance.json` 中 `screening_pass=true`；
- mAP50-95 相对 CA 至少 `+0.03`；
- AP75 不下降，AP90/AP95 下降均不超过 `0.002`；
- 短边和长边 boundary ratio 均不超过 `0.10`；
- 匹配 proposal 的平均 IoU 增量非负，改善比例不低于恶化比例；
- AMP 同配置复现与训练结果一致；
- FP32 batch=1 与 batch=8 的各项指标差值不超过 `5e-4`；
- CA SHA256 前后一致；
- test 未使用。

最后一条审计命令会自动核对三次复现是否使用同一个 CA 和 Refine 权重、运行模式是否正确、每次完整验收是否通过，以及 AMP 同配置复现和 FP32 batch=1/8 的差值是否都在 `5e-4` 以内。失败时脚本会保留 JSON/Markdown 证据并以非零状态退出。

完成后下载整个目录：

```text
/root/autodl-tmp/paper_exports/refine_v311_seed0
```

## 5. 内部范围锚点（不得写入论文）

- 记录日期：2026-08-07。
- 当前阶段将既有数据集及其划分视为固定且有效的实验前提，不把数据集问题纳入 Refine 结果判断、模型优化或后续实验决策。
- 论文正文、图表、实验分析和局限性部分均不引入数据集划分相关表述。
- 本条仅作为内部工作边界记录；除非用户后续明确重新开启该议题，否则不再主动分析或扩展。

## 6. 论文方法冻结锚点（2026-08-07）

- 当前方法结构冻结为 `Coverage-Aware Assignment + reg_max=32 + proposal-level geometry refine`。
- 正式论文结果固定采用独立验证的 FP32、batch=8、`imgsz=640` 口径；CA 权重固定为 `/root/autodl-tmp/work-dirs/yolo11_obb_640_811_ca/weights/best.pt`。
- 现有证据已经足以进入小论文写作：coarse/identity 完全恒等，CA 哈希不变，mAP50-95 为 `0.454137→0.562504`，AP75 为 `0.439822→0.558800`，匹配 proposal 平均 IoU 增量为 `+0.046647`。
- AP95 为 `0.099024→0.097174`，且当前仅有 seed0；论文采用“在当前固定实验设置下取得明确正向改善”的保守表述，不声称统计显著性。
- 后续只补统一 Baseline、H1/H2、复杂度、定性结果和必要的多种子复验。这些属于证据补全，不再因小幅数值波动触发 Refine 结构迭代。
