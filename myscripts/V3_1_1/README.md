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
BASELINE=/root/autodl-tmp/work-dirs/yolo11_obb_640_811_baseline/weights/best.pt
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

## 7. IVC 云端证据补全工具（2026-08-16）

固定路径：

```bash
export OMP_NUM_THREADS=1

BASELINE=/root/autodl-tmp/work-dirs/yolo11_obb_640_811_baseline/weights/best.pt
BASE=/root/autodl-tmp/work-dirs/yolo11_obb_640_811_ca/weights/best.pt
REFINE=/root/autodl-tmp/paper_exports/refine_v311_seed0/train_geometry_only/checkpoints/best.pt
DATA=/root/autodl-tmp/datasets/TTPLA-640-811/dataset.yaml
IVC_EXPORT=/root/autodl-tmp/paper_exports/ivc_evidence
```

### 7.1 完整 detector + NMS + Refine 性能采集

```bash
python -m myscripts.V3_1_1.profile_refine_v311 \
  --checkpoint "$REFINE" \
  --ca-weights "$BASE" \
  --data "$DATA" \
  --imgsz 640 \
  --batch 1 \
  --device 0 \
  --workers 8 \
  --no-amp \
  --warmup 20 \
  --output-dir "$IVC_EXPORT/profile_refine_fp32_batch1"
```

输出：

- `profile_per_image.csv`：逐图 proposal 数量、数据读取、预处理、CA forward、旋转 NMS、proposal 打包、Refine 和结果回写时延；
- `profile_summary.json`：各阶段 mean/median/P95、完整计算链 FPS、峰值显存、CA/Refine 参数量与权重哈希；正式延迟和显存窗口先于 THOP/`torch.profiler`，复杂度统计不会污染性能峰值；
- `refiner_profiled_gflops` 只统计 `torch.profiler` 能识别的 Refine 算子。`grid_sample` 和 NMS 可能没有 FLOPs 归因，因此完整计算成本以实测端到端时延为准。

正式口径锁定为 `val`、FP32、batch=1、`imgsz=640`。工具不会训练、修改或保存模型权重，也不执行第二次 NMS。

横向效率比较必须继续运行三轮平衡顺序工具：

```bash
python -m myscripts.V3_1_1.profile_comparative_v311 \
  --baseline-weights "$BASELINE" \
  --ca-weights "$BASE" \
  --refine-profile-summary "$IVC_EXPORT/profile_refine_fp32_batch1/profile_summary.json" \
  --data "$DATA" \
  --imgsz 640 --batch 1 --device 0 --workers 8 --no-amp \
  --warmup 20 --repeats 3 \
  --output-dir "$IVC_EXPORT/profile_comparative_fp32_batch1"
```

它按三阶拉丁方顺序启动9个独立子进程，消除模型间显存继承并平衡运行位置。除
`comparative_latency.csv`、`comparative_per_image.csv` 和审计 JSON 外，还输出
`comparative_repeat_summary.csv`。只有三轮延迟相对极差不超过5%、隔离峰值显存
相对极差不超过2%，且独立 CA 与 Refine 内部 coarse 的均值差不超过5%时，
`reportable_efficiency_pass` 才为 true。

### 7.2 GT / Baseline / CA / CA+Refine 逐图导出

```bash
python -m myscripts.V3_1_1.export_qualitative_v311 \
  --baseline-weights "$BASELINE" \
  --ca-weights "$BASE" \
  --checkpoint "$REFINE" \
  --data "$DATA" \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --no-amp \
  --copy-images \
  --output-dir "$IVC_EXPORT/qualitative_predictions"
```

输出目录包括 `images/`、`gt/`、`baseline/`、`ca/`、`refined/`、`manifest_all.csv` 和 `export_audit.json`。四套框被统一还原到原图坐标并保存为归一化四顶点 OBB；预测文件末列保留 confidence。导出器固定 Baseline 为 `reg_max=16`、CA 为 `reg_max=32`，并核对：

- Baseline、CA 和 Refine checkpoint 均为固定路径且哈希写入审计；
- 三种预测使用 checkpoint 中相同的 confidence、NMS IoU 和 `max_det`；
- Refine 不改变 proposal 数量、类别或分数，不执行第二次 NMS；
- 零残差回写与 CA proposal 逐坐标完全一致；
- `split=val`、`test_used=false`。

`manifest_all.csv` 包含完整验证集，不应直接全部绘图。下载完整输出目录后，按预先定义的场景类型选择代表性成功、一般和失败案例，把选中行保存为与 `manifest_all.csv` 同目录的 `fig6_manifest.csv`，填写同一 ROI 和客观 `note`。清单必须留在该目录中，才能继续按相对路径找到 `images/` 和四套标注。随后运行：

```powershell
python -m myscripts.paper_visuals.generate_fig6_qualitative `
  --manifest "res/640-811/ivc_evidence/qualitative_predictions/fig6_manifest.csv" `
  --show-confidence `
  --output-dir "mydocs/创新点一/投稿版本/IVC_assets"
```

### 7.3 IVC 全证据一键采集

综合入口会依次调用现有子脚本，完成环境与权重哈希记录、既有 holdout
证据复制、Baseline/CA 主结果、Refine 三协议复核、复现审计、复杂度、完整链
性能、四路定性导出以及 H1/H2 统计，最后生成逐文件 SHA256 和 `tar.gz`：

正式入口要求 Git 工作区干净，并把当前 commit 写入不可变运行身份；请先提交代码
再在云端采集。这样旧 commit 的结果不能通过 `--resume` 混入新证据包。

```bash
cd /root/Myultralytics
export OMP_NUM_THREADS=1

python -m myscripts.V3_1_1.collect_ivc_evidence
```

默认固定使用：

```text
Baseline: /root/autodl-tmp/work-dirs/yolo11_obb_640_811_baseline/weights/best.pt
CA:       /root/autodl-tmp/work-dirs/yolo11_obb_640_811_ca/weights/best.pt
Refine:   /root/autodl-tmp/paper_exports/refine_v311_seed0/train_geometry_only/checkpoints/best.pt
Data:     /root/autodl-tmp/datasets/TTPLA-640-811/dataset.yaml
```

未传 `--output-dir` 时会创建
`/root/autodl-tmp/paper_exports/ivc_evidence_YYYYMMDD_HHMMSS`。每个子任务的
标准输出单独写入 `logs/`，`run_state.json` 保存命令、用时、返回码和阶段状态。
若任务中断，使用控制台最后打印的目录继续：

```bash
python -m myscripts.V3_1_1.collect_ivc_evidence \
  --output-dir /root/autodl-tmp/paper_exports/ivc_evidence_YYYYMMDD_HHMMSS \
  --resume
```

续跑会锁定首次运行的数据、三个 checkpoint、训练证据目录、设备、workers 和
H1/H2 只读 passes 以及证据协议版本；其中任何一项变化都会拒绝继续，防止新旧
结果被混装。效率协议V2不能在旧协议证据目录上使用 `--resume`，必须创建新的
时间戳目录；V2目录自身中断后仍可正常续跑。

按需关闭耗时或重复步骤：

```bash
# 已经有 H1/H2 时不再运行两组统计循环
python -m myscripts.V3_1_1.collect_ivc_evidence --no-h1h2

# 跳过小样本 smoke，但仍执行两项正式完整采集
python -m myscripts.V3_1_1.collect_ivc_evidence --no-smoke

# 服务器没有 pytest 时跳过协议单元测试
python -m myscripts.V3_1_1.collect_ivc_evidence --no-tests

# 暂不压缩完整定性图片目录
python -m myscripts.V3_1_1.collect_ivc_evidence --no-archive
```

`H1/H2` 子脚本固定 checkpoint，在单一 `val` split 上执行 assigner 只读统计，
不计算梯度、不创建优化器、不执行训练/验证回调，也不保存 checkpoint；运行清单
会记录模型 state 前后哈希。综合入口还会生成 `profile_comparative_fp32_batch1/`
目录，使用三轮平衡顺序、独立进程和同一同步链路比较 Baseline、CA 与 CA+Refine；主验证与 Refine 验证 CSV
均输出 AP50、AP55、…、AP95。复现审计若仅因预声明的 `5e-4` 严格阈值返回非零，而
`reproduction_audit.json` 已完整写出且 `overall_pass=false`，综合入口会将该
阶段记为 `threshold_not_met` 并继续，不会把阈值未满足误报为采集器崩溃。
