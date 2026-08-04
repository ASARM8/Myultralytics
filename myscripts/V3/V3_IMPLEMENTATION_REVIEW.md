# Refine V3 数据结论、实现审核与可行性复核

## 1. 本轮结论

现有结果支持继续做 Refine，但不支持沿用 V1/V2.x 的 dense pointwise residual head。D2 已显示：P2/P3 旋转对齐 ROI 对“尺度修正”存在跨 train-fit、train-holdout 和 val 的稳定信号；中心偏移没有形成可靠信号。因此 V3 首版采用 post-NMS proposal 级、scale-only、质量门控的独立精修器，而不是同时预测中心、角度或修改分类分数。

这是一项 seed0 筛选方案。它的目标是验证“空间对齐特征能否转化为完整检测指标收益”，不是预先保证 Refine 一定优于 CA。

## 2. 已下载结果的证据

### 2.1 D1 proposal oracle

修正版 D1 v2 已使用统一口径完成。运行条件为 `imgsz=640`、`conf=0.01`、NMS IoU=0.70、max_det=300、oracle match IoU=0.30，权重为固定纯 CA checkpoint，数据划分为 val，未读取 test。

三项前置恒等性检查全部通过：

| 检查项 | 结果 | 判定 |
|---|---:|---|
| standard CA mAP50-95 | 0.4541379 | 与预期 0.45413 的绝对误差约 `7.9e-6`，PASS |
| top-K coarse mAP50-95 | 0.4541379 | 与 standard CA 六项指标完全一致，PASS |
| post-NMS coarse mAP50-95 | 0.4541379 | roundtrip 绝对误差为 0，PASS |

在这一有效基线上，pre-NMS oracle 相对 coarse 的变化如下：

| 变量 | mAP50-95 | 相对 coarse |
|---|---:|---:|
| scale | 0.784832 | +0.330694 |
| center | 0.491885 | +0.037747 |
| angle | 0.464110 | +0.009972 |
| scale+center | 0.898739 | +0.444601 |
| full geometry | 0.952698 | +0.498560 |

与 V3 实际作用位置一致的 post-NMS oracle 结果为：scale 0.787313（+0.333175）、center 0.492989（+0.038852）、scale+center 0.898685（+0.444547）、full geometry 0.951378（+0.497240）。其中 scale-only 在 AP75/AP90/AP95 上分别达到约 0.881252/0.546571/0.229690，说明尺度仍是最主要的可修正几何因素，并且该上限在重新 NMS 后没有消失。

pre-NMS top-K 的 GT recall 在 IoU 0.50/0.75/0.90 下分别为 0.834582/0.615598/0.423536；post-NMS 后分别为 0.822648/0.576742/0.351096。它一方面证明保留 proposal 中仍有较大的定位优化空间，另一方面明确了 V3 的边界：post-NMS Refine 不能恢复已经漏掉或被 NMS 删除的候选框，只能改善保留下来的候选框定位。

上述 oracle 是把 proposal 几何直接替换为匹配 GT 后得到的理论上限，不是可学习模型成绩，也不能作为论文中的实际精度。它只用于支持“优先训练 scale-only Refine”的结构选择。旧 `conf=0.001`、mAP50-95=0.421885 的 D1 结果口径无效，仅保留为问题追溯记录，不再用于后续决策。

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

### 2.4 seed0 与独立复现

seed0 在 train-holdout 选择 epoch15、quality threshold=0.3。AMP/batch=8 的 val coarse/refined mAP50-95 为 0.453562/0.695961（Δ=+0.242399），AP75、AP90 分别提高 0.347656、0.096331；gate ratio 为 0.739932。重新加载 `best.pt` 的独立验证产出与首次 val 的指标和残差 JSON 逐字节一致，排除了 checkpoint 保存/加载、阈值恢复和训练内存态导致的偶然结果。

FP32/batch=1 进一步得到 coarse/refined=0.454151/0.698983（Δ=+0.244832），roundtrip 仍严格恒等。相对 AMP/batch=8，refined 绝对值变化约 +0.00302，其中 CA baseline 变化约 +0.00059。收益方向和量级稳定，但变化超过此前建议的 5e-4 数值阈值，因此正式结果固定为 FP32，并补做 FP32/batch=8 与 AMP/batch=1 的 2×2 精度/批大小审计。

当前结果已经证明冻结 checkpoint 在现有 V3 评估链中稳定有效，但尚不能单凭聚合 mAP 证明实例级 ROI、quality gate 和 re-NMS 分别必要。为排除全局短边校准、proposal 错配和数据重复等替代解释，已新增完整机制真实性审计和 train/val 图像重复审计；在这些结果返回前，`+0.242～0.245` 仍按强 seed0 证据而非正式论文结论处理。

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
| seed0 大幅收益可能是统一缩放捷径 | 已新增审计 | 固定 holdout 均值残差、all-refine、short/long-only 对照 |
| residual/quality 与 proposal 对应可能无关 | 已新增审计 | 图像内 residual/quality shuffle，保持边际分布并破坏对应关系 |
| ROI 空间特征可能没有实际贡献 | 已新增审计 | 打乱有效 proposal 的 ROI 编码，同时保持 proposal state 不变 |
| re-NMS 可能解释主要收益 | 已新增审计 | 同一预测输出比较 selected gate 与 selected-no-reNMS |
| train/val 图像重复 | 已新增审计 | 完整路径、SHA256、dHash 近重复与人工拼图；test 不读取 |

另外，train-fit 为便于随机打乱和任意分组子集，使用无增强的方形 640 letterbox；val 按标准 rect 模式评估。该差异不会改变标签坐标一致性，但可能形成轻微域差异，已列为 seed0 后的观察项。若 seed0 出现“holdout 正向、val 反向”，优先复核该项和场景分组，而不是直接扩大网络。

## 5. 再次复核：合理性与可行性

### 合理性

1. **监督对象与推理对象一致**：V3 在 post-NMS proposal 上训练和推理，不再用 dense anchor 监督替代最终候选框行为。
2. **自由度与证据一致**：scale 有稳定信号，center 没有，因此首版只开放两个尺度自由度。
3. **残差范围与数据一致**：短边强负偏、长边近零，非对称范围比统一小范围更符合目标分布。
4. **安全退化路径完整**：几何零初始化、quality gate、identity 正则、NMS roundtrip 和 CA baseline 校验共同保证问题可定位。
5. **评价边界合理**：epoch/阈值不使用 val，test 继续封存，避免把单验证集后验调参写成模型学习收益。

### 可行性

修正版 D1、seed0 训练、checkpoint 独立复现和 FP32/batch=1 稳定性检查均已通过，说明尺度方向不仅存在 oracle 上限，而且已转化为很强的单 checkpoint val 收益。当前主要不确定性已经收敛为机制归因与数据完整性：实例级 ROI 是否优于固定尺度校准、quality gate 是否真正识别受益 proposal、re-NMS 贡献多大，以及 train/val 是否存在视觉近重复。

因此当前不扩张结构，先运行已实现的真实性审计。结果将区分后续方向：

- 若尺度回归诊断正向但 gate 较差：优先改 quality target/采样与阈值学习。
- 若匹配样本损失下降但 residual 近常数：增加 proposal jitter、分层条件或 hard-example sampling。
- 若 holdout 正向而 val 反向：先查场景分组、方形/rect 差异和匹配分布偏移。
- 若 post-NMS oracle 本身很低：转向 pre-NMS learned rescoring/refinement，而不是继续堆 post-NMS head。
- 若 seed0 达到筛选门槛：再进行独立 CA seed 的配对复核；在此之前不读取 test。

## 6. seed0 验收标准

硬前置条件：

- corrected D1 的 standard CA mAP50-95 在 `0.45413±0.002`：已通过（0.4541379）；
- post-NMS roundtrip 与 coarse 的 mAP50-95 差值不超过 `5e-4`：已通过（差值 0）；
- CA SHA256、head 类型和 `reg_max` 检查全部通过：已通过。

seed0 初筛门槛：

- refined 相对 coarse 的 mAP50-95 至少 +0.002；
- AP75 不下降；
- AP90 下降不超过 0.002；
- 残差不退化为明显层级/全局常数，门控比例不是 0% 或 100%；
- train-holdout 与 val 的方向一致。

未达门槛不等于主动终止 Refine。应按第5节的故障分型进入 V3.1；只有在用户决定停止时，才结束该研究路线。

## 7. 执行顺序

1. 完整 V3 `python -m pytest`：已完成。
2. 一个 train batch 的 `smoke_refine_v3.py`：已完成并通过真实接口闭环。
3. corrected D1 v2：已完成；同口径 CA baseline、top-K identity、post-NMS roundtrip 和 scale oracle 均通过。
4. seed0 训练与冻结后一次 val：已完成，筛选通过。
5. `best.pt` 独立复现：已完成，输出逐字节一致。
6. FP32/batch=1：已完成，收益方向与量级稳定；发现约 0.003 的精度/批大小联合差异。
7. 下一步运行 FP32/batch=8 与 AMP/batch=1，拆分精度和 batch 影响。
8. 运行 `audit_refine_v3.py` 完成机制对照与逐 proposal/subgroup 审计。
9. 运行 `audit_dataset_splits_v3.py` 完成 train/val 精确及感知近重复检查。
10. 上述审计通过后才进入独立 CA seed 配对训练；test 继续封存。

完整命令见同目录 `README.md`。
