# Refine V3 数据结论、实现审核与可行性复核

## 1. 本轮结论

现有结果支持继续做 Refine，但不支持沿用 V1/V2.x 的 dense pointwise residual head。D2 已显示：P2/P3 旋转对齐 ROI 对“尺度修正”存在跨 train-fit、train-holdout 和 val 的稳定信号；中心偏移没有形成可靠信号。因此 V3 首版采用 post-NMS proposal 级、scale-only、质量门控的独立精修器，而不是同时预测中心、角度或修改分类分数。

这是一项 seed0 筛选方案。它的目标是验证“空间对齐特征能否转化为完整检测指标收益”，不是预先保证 Refine 一定优于 CA。

## 2. 已下载结果的证据

### 2.1 D1 proposal oracle

旧 D1 的 `standard_ca` 与 `topk_coarse` 完全一致，mAP50-95 都为 0.421885；pre-NMS oracle 相对该基线的变化如下：

| 变量 | mAP50-95 | 相对 coarse |
|---|---:|---:|
| scale | 0.769248 | +0.347364 |
| center | 0.458432 | +0.036548 |
| angle | 0.431152 | +0.009267 |
| scale+center | 0.890063 | +0.468178 |
| full geometry | 0.950014 | +0.528130 |

pre-NMS top-K 的 GT recall 为：IoU 0.50/0.75/0.90 下分别为 0.845129/0.623647/0.428254。它说明候选集合含有大量可利用几何信息，也说明高 IoU recall 仍是上限约束。

但旧 D1 显式使用 `conf=0.001`，当前仓库 OBB val 默认是 `conf=0.01`，而训练记录中的 CA 最佳 mAP50-95 为约 0.45413。因此 0.421885 不是同口径基线，不能直接用于正式决策。代码已将默认值改为 0.01，并加入 `0.45413±0.002` 硬校验；还新增 post-NMS oracle，因为正式 V3 处理的是 post-NMS proposal。

### 2.2 D2 rotated ROI probe

有效匹配样本：train 共收集 28,552 个并保留 20,000 个；val 为 3,484 个。train 以图像组拆为 8,886 个 fit group 和 2,221 个 holdout group，组重叠为 0。

| 特征 | val scale MAE 改善 | val 方向准确率 | val ΔProbIoU | 改善/恶化比例 |
|---|---:|---:|---:|---:|
| proposal state | 0.31617 | 0.67567 | +0.04809 | 63.20% / 36.80% |
| P2/P3 rotated ROI | 0.35792 | 0.70704 | +0.05621 | 68.11% / 31.86% |

ROI scale 的 train-fit、train-holdout、val ΔProbIoU 分别约为 +0.06145、+0.06058、+0.05621，差距较小，暂未表现为单纯记忆训练样本。ROI 比 state 在 val 上再增加约 0.0081 ProbIoU，支持保留空间对齐条带。

中心分支在 val 上的 MAE 改善为负，ΔProbIoU 约为 -0.000032，不能证明可学习。因此中心预测默认禁用，只保留为后续显式消融开关。

### 2.3 D3 640 分辨率审计

val 共 3,603 个实例。短边 P05/P50/P95 约为 3.61/10.83/32.82 px；短边小于 4/8/16/32 px 的比例约为 7.72%/33.33%/71.55%/94.42%。长宽比 P50/P95 约为 35.74/137.92。

在 val 中，1 px 扰动的平均 `1-ProbIoU`：短轴中心移动约 0.14595、短边增加约 0.05975、端点诱导角度变化约 0.08682。说明极细目标对像素量化敏感。实现中不硬删除 tiny 样本，而是按 `clamp(short/8, 0.25, 1)` 平滑降权，避免少量离散标注噪声主导尺度损失。

尺度目标分布也明显非对称：`dshort` 均值约 -0.408、P05 约 -1.487、P95 约 +0.151；`dlong` 大多集中在 0 附近。V3 因此使用短边 `[-1.5,+0.25]`、长边 `[-0.15,+0.15]`，而不是对称 `±0.5`。

## 3. 已实现的正式 V3 定义

### 3.1 数据流

1. 加载并验证固定纯 CA checkpoint：检测头必须正好是 `OBB`，`reg_max=32`。
2. CA 始终为 `eval()`，全部参数 `requires_grad=False`；P2 和 P3 特征通过只读 hook 提取。
3. 使用 `conf=0.01`、NMS IoU=0.70、max_det=300 得到 post-NMS proposal。
4. 以类别一致且 ProbIoU≥0.30 为候选，按 IoU 从高到低执行一对一 greedy matching。
5. 从 P2/P3 为每个 proposal 提取长轴对齐的旋转条带，并与置信度、尺度、位置、角度状态融合。
6. 预测短边/长边对数尺度残差和独立 quality logit；类别、分类分数和角度不变。
7. quality 达到 train-holdout 已选阈值时才应用尺度残差，随后重新运行 rotated NMS。

### 3.2 损失与门控

- 几何监督：匹配 proposal 到等价表示下的 GT 长短边，使用加权 SmoothL1。
- 有界映射：按残差符号使用不同 `tanh` 范围，在 0 点函数值为 0、导数为 1。
- target margin：监督目标限制在推理边界的 99% 内，避免要求 `tanh` 达到不可达的精确端点。
- quality 正样本：有界尺度 oracle 相对 coarse 的 ProbIoU 至少增加 0.002。
- quality 损失：二元 focal loss，用于处理可精修 proposal 与普通/未匹配 proposal 的不平衡。
- identity 正则：quality 阴性和未匹配 proposal 的尺度残差轻量趋零。
- quality 只控制几何是否生效，不乘入或改写 CA 分类置信度。
- 优化器：AdamW、`lr=3e-4`，前3轮线性 warmup，只更新 V3 参数。

### 3.3 数据选择边界

- train 按稳定哈希拆为 train-fit 和 train-holdout；若存在连续帧，应通过 `--group-regex` 按场景拆分。
- epoch 和 quality threshold 只根据 train-holdout 的完整 mAP50-95 选择。
- 只要存在非退化候选，选择时排除 gate ratio≤0.1% 或≥99.9% 的阈值，避免把“全部关闭/全部开启”误选为有效质量门控。
- 选择冻结之后才构造 val loader，并只运行一次 val。
- test 不开放、不读取、不用于任何结构、阈值或 epoch 选择。

## 4. 代码审核记录

| 审核项 | 结果 | 处理 |
|---|---|---|
| CA 路径漂移 | 已防护 | 训练/验证入口锁定用户指定 canonical 路径 |
| 错载旧 CA+Refine | 已防护 | head 类型必须为纯 `OBB`，且 `reg_max=32` |
| 同名 CA 文件内容变化 | 已防护 | checkpoint 保存 CA SHA256；复现验证要求哈希完全一致 |
| CA 参数或 BN 漂移 | 已防护 | CA 全冻结并保持 eval，仅独立 V3 进入 optimizer |
| 冻结特征无法供 V3 反向传播 | 已修复 | CA 前向使用 `torch.no_grad()`，不用会产生 inference tensor 的 `inference_mode()` |
| 新头初始破坏 coarse | 已防护 | geometry head 零初始化；roundtrip mAP 容差为 5e-4 |
| 验证置信度口径不一致 | 已修复 | D1、trainer、validator 统一为 0.01并校验 CA mAP |
| 单类别 OBB 的 `[1,6,6]` NMS 形状碰撞 | 已修复 | K=6 时追加零置信度哨兵列，并对 post-NMS coarse identity 做硬检查 |
| V3 作用位置与 oracle 不一致 | 已修复 | D1 新增 post-NMS coarse 与 post-NMS 几何 oracle |
| OBB 宽高/角度等价歧义 | 已防护 | 监督前选择最近等价 GT；状态方向改用等价表示不变的长轴方向 |
| 多 proposal 重复匹配同一 GT | 正式训练已防护 | 类别感知、一对一 greedy matching；D1 oracle仍是上限分析 |
| tanh 边界不可达 | 已修复 | target margin=0.99 |
| 布尔索引通道歧义 | 已修复 | 先执行二维 valid mask，再切残差通道 |
| tiny 标注噪声 | 已缓解 | 连续降权，不硬删除 |
| 中心分支无证据仍参与训练 | 已避免 | `enable_center=False`，输出中心残差恒为0 |
| padding proposal 污染 BatchNorm | 已避免 | V3 使用按样本计算的 GroupNorm；无效 padding 不改变其他 proposal 的统计量 |
| 质量阈值后验使用 val | 已避免 | 阈值仅在 train-holdout 选择 |
| test 泄漏 | 已避免 | 参数 choices 不提供 test |
| 本地无 torch/pytest | 预期环境差异 | 重型导入延迟到 main；本地完成 AST/`--help`，完整测试在云端运行 |
| 静态测试未覆盖真实数据接口 | 已补充 | `smoke_refine_v3.py` 用一个 train batch 检查 hook、匹配、identity、反向传播和 CA 无梯度 |

另外，train-fit 为便于随机打乱和任意分组子集，使用无增强的方形 640 letterbox；val 按标准 rect 模式评估。该差异不会改变标签坐标一致性，但可能形成轻微域差异，已列为 seed0 后的观察项。若 seed0 出现“holdout 正向、val 反向”，优先复核该项和场景分组，而不是直接扩大网络。

## 5. 再次复核：合理性与可行性

### 合理性

1. **监督对象与推理对象一致**：V3 在 post-NMS proposal 上训练和推理，不再用 dense anchor 监督替代最终候选框行为。
2. **自由度与证据一致**：scale 有稳定信号，center 没有，因此首版只开放两个尺度自由度。
3. **残差范围与数据一致**：短边强负偏、长边近零，非对称范围比统一小范围更符合目标分布。
4. **安全退化路径完整**：几何零初始化、quality gate、identity 正则、NMS roundtrip 和 CA baseline 校验共同保证问题可定位。
5. **评价边界合理**：epoch/阈值不使用 val，test 继续封存，避免把单验证集后验调参写成模型学习收益。

### 可行性

实现可以直接进入云端 D1 修正版和 seed0 训练。D2 的跨分组结果表明该方向具有可学习性；scale oracle 上限也足够大。主要不确定性不在“有没有尺度信号”，而在三个转换环节：ROI 回归是否能在完整 proposal 分布上保持收益、quality gate 能否识别净受益 proposal、微小几何改善能否在重新 NMS 后转化为 mAP。

因此当前最小可行实验不是再扩张结构，而是先运行已实现的闭环。seed0 结果将区分后续方向：

- 若尺度回归诊断正向但 gate 较差：优先改 quality target/采样与阈值学习。
- 若匹配样本损失下降但 residual 近常数：增加 proposal jitter、分层条件或 hard-example sampling。
- 若 holdout 正向而 val 反向：先查场景分组、方形/rect 差异和匹配分布偏移。
- 若 post-NMS oracle 本身很低：转向 pre-NMS learned rescoring/refinement，而不是继续堆 post-NMS head。
- 若 seed0 达到筛选门槛：再进行独立 CA seed 的配对复核；在此之前不读取 test。

## 6. seed0 验收标准

硬前置条件：

- corrected D1 的 standard CA mAP50-95 在 `0.45413±0.002`；
- post-NMS roundtrip 与 coarse 的 mAP50-95 差值不超过 `5e-4`；
- CA SHA256、head 类型和 `reg_max` 检查全部通过。

seed0 初筛门槛：

- refined 相对 coarse 的 mAP50-95 至少 +0.002；
- AP75 不下降；
- AP90 下降不超过 0.002；
- 残差不退化为明显层级/全局常数，门控比例不是 0% 或 100%；
- train-holdout 与 val 的方向一致。

未达门槛不等于主动终止 Refine。应按第5节的故障分型进入 V3.1；只有在用户决定停止时，才结束该研究路线。

## 7. 执行顺序

1. 运行完整 V3 `python -m pytest`。
2. 运行一个 train batch 的 `smoke_refine_v3.py`，确认云端真实接口闭环。
3. 运行 corrected D1，确认同口径 CA baseline 与 post-NMS scale oracle。
4. 运行 `train_refine_v3.py` 完成 seed0 训练、holdout 选择和一次 val。
5. 下载整个输出目录，重点提供 `run_manifest.json`、`train_history.csv`、`holdout_metrics.csv`、`selection.json`、`val_metrics.csv`、`val_diagnostics.json` 和 `acceptance.json`。
6. `validate_refine_v3.py` 仅用于复现审核，不用于重新选择阈值。

完整命令见同目录 `README.md`。
